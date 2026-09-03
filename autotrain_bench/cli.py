"""Command-line interface for users and containerized agents."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .agent_api_bundle import publish_agent_api_bundle
from .challenge import ChallengeRelease, ResolvedChallengeRun
from .client import Client, ClientError
from .config import (
    AgentConfig,
    ConfigError,
    InferenceConfig,
    TaskConfig,
    TrainingResources,
    agent_egress,
    agent_environment_names,
    agent_runner,
)
from .host_agent import (
    HostAgentProcess,
    HostAgentStartupFailure,
    summarize_host_agent_trace,
)
from .inference import summarize_inference_trace
from .local_mailbox import LocalFileMailbox
from .local_agent_execution import describe_local_agent_execution
from .results import normalize_result_report
from .runtime import DockerRuntime, RuntimeFailure
from .supervisor import Supervisor


CONTAINER_AGENT_DRAIN_SECONDS = 3.0
HOST_AGENT_DRAIN_SECONDS = 30.0


def _print(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _episode_directory(root: Path) -> Path:
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (root / f"{timestamp}-{uuid.uuid4().hex[:8]}").resolve()


def _find_git_root(path: Path) -> Optional[Path]:
    """Return the enclosing Git checkout without invoking a site-specific tool."""

    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _trusted_workspace_roots(task_source: Path) -> List[Path]:
    """Return trusted Core/task checkouts that may not host Agent workspaces."""

    roots = {Path(__file__).resolve().parent.parent}
    task_checkout = _find_git_root(task_source.parent)
    roots.add(task_checkout or task_source.parent.resolve())
    return sorted(roots, key=str)


def _validate_workspace_boundary(
    *,
    workspace: Path,
    task_source: Path,
    agent_workspace: Optional[Path],
) -> None:
    """Fail before task reveal when trusted files could enter Agent scope."""

    resolved_workspace = workspace.resolve()
    trusted_roots = _trusted_workspace_roots(task_source.resolve())
    for root in trusted_roots:
        if _is_within(resolved_workspace, root) or _is_within(root, resolved_workspace):
            raise ConfigError(
                "episode workspace must be outside the trusted Core/task checkout "
                "and must not contain it: "
                f"{resolved_workspace} overlaps {root}; choose a dedicated external "
                "--episode-dir or --episodes-root"
            )

    if agent_workspace is None:
        return
    resolved_source = agent_workspace.resolve()
    for root in trusted_roots:
        if _is_within(resolved_source, root) or _is_within(root, resolved_source):
            raise ConfigError(
                "--agent-workspace must be participant-owned and outside the trusted "
                "Core/task checkout, and must not contain it: "
                f"{resolved_source} overlaps {root}"
            )


def _snapshot_agent_workspace(source: Path, destination: Path) -> None:
    resolved_source = source.resolve()
    resolved_destination = destination.resolve()
    try:
        resolved_destination.relative_to(resolved_source)
    except ValueError:
        pass
    else:
        raise ConfigError("episode workspace may not be inside the source workspace")
    try:
        shutil.copytree(resolved_source, resolved_destination, symlinks=True)
    except OSError as exc:
        raise ConfigError(f"failed to snapshot agent workspace: {exc}") from exc


def _initialize_agent_workspace(source: Optional[Path], destination: Path) -> None:
    """Create an empty episode workspace or snapshot explicit participant files."""

    if source is None:
        destination.mkdir()
    else:
        _snapshot_agent_workspace(source.resolve(), destination)


def _terminal_agent_action(
    *, runner: str, running: bool, drain_elapsed_seconds: float
) -> str:
    """Return whether a terminal episode should keep draining its Agent."""

    if not running:
        return "exited"
    limit = (
        HOST_AGENT_DRAIN_SECONDS
        if runner == "host_cli"
        else CONTAINER_AGENT_DRAIN_SECONDS
    )
    return "drain_timeout" if drain_elapsed_seconds >= limit else "wait"


def _episode_exit_code(state: Dict[str, Any]) -> int:
    """Return success only for an episode with a successful TEST outcome."""

    if state.get("status") != "finalized" or state.get("outcome") != "success":
        return 1
    if state.get("active_lease") is not None:
        return 1
    if state.get("pending_lease_teardowns"):
        return 1
    if state.get("deployment_teardown_failures"):
        return 1
    agent_control = state.get("agent_control")
    if (
        isinstance(agent_control, dict)
        and agent_control.get("cleanup_complete") is not True
    ):
        return 1
    return 0


CleanupFailure = Tuple[str, str]
CleanupStep = Tuple[str, Callable[[], Any]]


def _run_cleanup_steps(steps: Sequence[CleanupStep]) -> List[CleanupFailure]:
    """Attempt every trusted deployment cleanup and retain private failures."""

    failures = []
    for code, cleanup in steps:
        try:
            result = cleanup()
        except (Exception, KeyboardInterrupt) as exc:
            failures.append((code, f"{type(exc).__name__}: {exc}"))
        else:
            if result is False:
                failures.append((code, "cleanup reported incomplete"))
    return failures


def _report_cleanup_failures(
    supervisor: Supervisor, failures: Sequence[CleanupFailure]
) -> None:
    """Persist deployment-integrity failures independently of metric outcome."""

    if not failures:
        return
    reporting_errors = []
    for code, private_detail in failures:
        try:
            supervisor.deployment_teardown_failed(code, private_detail=private_detail)
        except (Exception, KeyboardInterrupt) as exc:
            reporting_errors.append(exc)
    if reporting_errors:
        # Every failure was offered to the authoritative state machine before
        # returning the first persistence/reporting error to the operator.
        raise reporting_errors[0]


def _stop_and_record_container_agent(
    *,
    runtime: DockerRuntime,
    supervisor: Supervisor,
    name: str,
    drain_seconds: float,
) -> None:
    """Record termination only after the local runtime confirms Agent removal."""

    exit_code = None
    termination_reason = "metadata_unavailable"
    try:
        running = runtime.container_running(name)
        exit_code = None if running else runtime.container_exit_code(name)
        termination_reason = "drain_timeout" if running else "exited"
    except RuntimeFailure:
        pass
    runtime.stop_container(name)
    supervisor.record_agent_execution(
        exit_code=exit_code,
        termination_reason=termination_reason,
        drain_seconds=drain_seconds,
    )


def _stop_and_record_host_agent(
    *,
    supervisor: Supervisor,
    host_agent: HostAgentProcess,
    termination_reason: str,
    drain_seconds: float,
) -> None:
    """Record host-Agent termination only after its process has stopped."""

    exit_code = host_agent.stop()
    usage = summarize_host_agent_trace(host_agent.stdout_path)
    supervisor.record_agent_execution(
        exit_code=exit_code,
        usage=usage,
        termination_reason=termination_reason,
        drain_seconds=drain_seconds,
    )


def _stop_and_record_startup_container_agent(
    *,
    runtime: DockerRuntime,
    supervisor: Supervisor,
    name: str,
    termination_reason: str = "episode_startup_failure",
) -> None:
    """Confirm an uncertain startup workload is absent before recording it."""

    runtime.stop_container(name)
    supervisor.record_agent_execution(
        exit_code=None,
        termination_reason=termination_reason,
    )


def command_prepare(args: argparse.Namespace) -> int:
    config = TaskConfig.load(args.task)
    reference_runtime = DockerRuntime()
    reference_runtime.prepare(config, pull_missing=not args.no_pull)
    worker_environment = reference_runtime.environment_manifest(config)
    _print(
        {
            "status": "ready",
            "task_id": config.task_id,
            "task_kind": config.task_kind,
            "leaderboard_eligible": config.leaderboard_eligible,
            "images": config.images,
            "evaluator_backend": config.evaluator_backend,
            "evaluator_policy_compliant": config.evaluator_policy_compliant,
            "worker": reference_runtime.describe_capabilities().public_manifest(),
            "worker_environment": worker_environment,
        }
    )
    return 0


def _resolve_challenge_run(
    args: argparse.Namespace,
    config: TaskConfig,
    release: Optional[ChallengeRelease] = None,
) -> Optional[ResolvedChallengeRun]:
    """Resolve the named leaderboard lane, or validate an ad-hoc task run."""

    if args.challenge is None:
        if args.track_config or args.runtime_profile:
            raise ConfigError(
                "--track-config and --runtime-profile require --challenge"
            )
        if not args.track:
            raise ConfigError("--track is required when --challenge is not used")
        return None
    if args.track:
        raise ConfigError(
            "--track is selected by --track-config when --challenge is used"
        )
    if not args.track_config or not args.runtime_profile:
        raise ConfigError("--challenge requires --track-config and --runtime-profile")
    if args.training_gpus or args.training_gpu_model or args.training_devices:
        raise ConfigError(
            "manual training resources may not override a challenge runtime profile"
        )
    release = release or ChallengeRelease.load(args.challenge)
    return release.resolve(
        config,
        track_config_id=args.track_config,
        runtime_profile_id=args.runtime_profile,
    )


def command_start(args: argparse.Namespace) -> int:
    if getattr(args, "agent_mode", None) is not None:
        raise ConfigError(
            "--agent-mode was renamed to --agent-egress and its values changed: "
            "'sandboxed' is now 'denied'; 'unrestricted-dev' is unchanged"
        )
    release = ChallengeRelease.load(args.challenge) if args.challenge else None
    if release is None:
        if args.task is None:
            raise ConfigError("--task is required when --challenge is not used")
        task_path = args.task
    else:
        task_path = release.task_path
        if args.task is not None and args.task.expanduser().resolve() != task_path:
            raise ConfigError("--task may not override the task owned by --challenge")
    config = TaskConfig.load(task_path)
    resolved_challenge = _resolve_challenge_run(args, config, release)
    comparison_key = None
    challenge_release = None
    if resolved_challenge is None:
        track = args.track
        resources = TrainingResources(
            nvidia_gpu_count=args.training_gpus,
            required_gpu_model=args.training_gpu_model,
            devices=args.training_devices,
        ).validate_for(config)
    else:
        config = resolved_challenge.task
        track = resolved_challenge.track
        resources = resolved_challenge.resources
        comparison_key = resolved_challenge.comparison_key.public_manifest()
        challenge_release = resolved_challenge.release_manifest
    resolved_runner = agent_runner(args.agent_runner, "--agent-runner")
    if resolved_runner == "host_cli":
        if args.agent_egress:
            raise ConfigError("--agent-egress is not valid with host_cli")
        resolved_egress = None
    else:
        resolved_egress = agent_egress(args.agent_egress or "denied", "--agent-egress")
    command = shlex.split(args.agent_command)
    environment = agent_environment_names(args.agent_environment, "--agent-env")

    inference = None
    if args.inference_base_url:
        inference = InferenceConfig.load(
            {
                "base_url": args.inference_base_url,
                "api_key_env": args.inference_api_key_env,
                "allowed_models": args.inference_models,
                "allowed_paths": args.inference_paths
                or ["/v1/responses", "/v1/chat/completions"],
                "timeout_seconds": args.inference_timeout,
            },
            "inference",
        )
    elif args.inference_api_key_env or args.inference_models or args.inference_paths:
        raise ConfigError(
            "--inference-base-url is required when configuring registered inference"
        )

    episode_dir = (
        args.episode_dir.resolve()
        if args.episode_dir
        else _episode_directory(args.episodes_root)
    )
    workspace = episode_dir / "workspace"
    _validate_workspace_boundary(
        workspace=workspace,
        task_source=config.source,
        agent_workspace=args.agent_workspace,
    )
    episode_dir.mkdir(parents=True, exist_ok=False)
    try:
        _initialize_agent_workspace(args.agent_workspace, workspace)
        agent_api_dir = publish_agent_api_bundle(episode_dir / "public" / "agent_api")
        agent_control = LocalFileMailbox(episode_dir / "control")
        agent_control.initialize()
        agent = AgentConfig(
            workspace=workspace,
            runner=resolved_runner,
            command=command,
            image=args.agent_image,
            egress=resolved_egress,
            environment=environment,
            inference=inference,
            initial_workspace=(
                "empty" if args.agent_workspace is None else "participant_snapshot"
            ),
        ).validate()
        agent_execution = describe_local_agent_execution(agent)
        missing_environment = [
            name for name in agent.environment if name not in os.environ
        ]
        if missing_environment:
            names = ", ".join(missing_environment)
            raise ConfigError(f"agent environment variable is not set: {names}")
        # Local file locations are private composition inputs for the bundled
        # Worker.  They never cross the semantic Core/Worker protocol, and may
        # remain absent until post-research evaluation begins.
        reference_runtime = DockerRuntime(
            hidden_datasets={
                "analysis": config.analysis_path,
                "test": config.test_path,
            }
        )
        reference_runtime.prepare(
            config,
            agent=agent,
            resources=resources,
            pull_missing=False,
        )
        worker_environment = reference_runtime.environment_manifest(
            config,
            agent=agent,
            resources=resources,
        )
    except Exception:
        shutil.rmtree(episode_dir, ignore_errors=True)
        raise

    supervisor = Supervisor(
        config=config,
        agent=agent,
        resources=resources,
        track=track,
        episode_dir=episode_dir,
        agent_api_dir=agent_api_dir,
        agent_execution=agent_execution,
        research_worker=reference_runtime,
        evaluation_worker=reference_runtime,
        agent_control_transport=agent_control,
        research_worker_environment=worker_environment,
        evaluation_worker_environment=worker_environment,
        comparison_key=comparison_key,
        challenge_release=challenge_release,
    )
    runtime_id = uuid.uuid5(uuid.NAMESPACE_URL, str(episode_dir)).hex[:12]
    prefix = f"atb-{runtime_id}"
    agent_name = f"{prefix}-agent"
    gateway_name = f"{prefix}-inference"
    agent_network = f"{prefix}-agent-net"
    egress_network = f"{prefix}-egress-net"
    agent_started = False
    host_agent: Optional[HostAgentProcess] = None
    gateway_started = False
    networks_to_cleanup: List[str] = []
    print(f"Episode: {episode_dir}", flush=True)
    try:
        if agent.inference is not None:
            # A runtime create call may time out after creating the resource.
            # Register each deterministic name before the attempt so startup
            # teardown can remove a resource whose creation result is unknown.
            networks_to_cleanup.append(agent_network)
            reference_runtime.create_network(
                agent_network,
                internal=agent.egress == "denied",
            )
            networks_to_cleanup.append(egress_network)
            reference_runtime.create_network(egress_network, internal=False)
            reference_runtime.start_inference_gateway(
                name=gateway_name,
                agent_network=agent_network,
                egress_network=egress_network,
                config=config,
                agent=agent,
                audit_dir=supervisor.inference_dir,
            )
            gateway_started = True
        supervisor.begin()
        if agent.runner == "container":
            reference_runtime.start_agent(
                name=agent_name,
                config=config,
                agent=agent,
                control_dir=agent_control.control_dir,
                public_task_path=supervisor.public_task_path,
                inference_network=agent_network
                if agent.inference is not None
                else None,
            )
        else:
            host_agent = reference_runtime.start_host_agent(
                agent=agent,
                control_dir=agent_control.control_dir,
                public_task_path=supervisor.public_task_path,
                stdout_path=episode_dir / "agent.stdout.log",
                stderr_path=episode_dir / "agent.stderr.log",
            )
        agent_started = True
    except BaseException as exc:
        # Once startup crosses a deployment side-effect boundary, every exit
        # path owns compensation.  In particular, Ctrl-C must not bypass the
        # cleanup that exists before the steady-state ``finally`` is installed.
        startup_error = exc
        if isinstance(exc, HostAgentStartupFailure):
            # Popen succeeded, so ``None`` would be an unsafe statement about
            # process ownership.  Preserve the retryable handle even when the
            # launch helper's first termination attempt also failed.
            host_agent = exc.handle
            startup_error = exc.startup_error
        termination_reason = (
            "supervisor_interrupted"
            if isinstance(startup_error, KeyboardInterrupt)
            else "episode_startup_failure"
        )
        private_detail = f"{type(startup_error).__name__}: {startup_error}"
        if isinstance(exc, HostAgentStartupFailure) and exc.cleanup_error is not None:
            private_detail += (
                "; initial host Agent cleanup failed: "
                f"{type(exc.cleanup_error).__name__}: {exc.cleanup_error}"
            )

        transition_error: Optional[BaseException] = None
        try:
            supervisor.infrastructure_failed(
                termination_reason,
                private_detail=private_detail,
            )
        except BaseException as transition_exc:
            transition_error = transition_exc

        startup_cleanup_steps: List[CleanupStep] = []
        if agent.runner == "container":
            startup_cleanup_steps.append(
                (
                    "agent_termination_failure",
                    lambda: _stop_and_record_startup_container_agent(
                        runtime=reference_runtime,
                        supervisor=supervisor,
                        name=agent_name,
                        termination_reason=termination_reason,
                    ),
                )
            )
        else:
            if host_agent is not None:

                def record_host_startup() -> None:
                    _stop_and_record_host_agent(
                        supervisor=supervisor,
                        host_agent=host_agent,
                        termination_reason=termination_reason,
                        drain_seconds=0.0,
                    )

            else:

                def record_host_startup() -> None:
                    supervisor.record_agent_execution(
                        exit_code=None,
                        termination_reason=termination_reason,
                    )

            startup_cleanup_steps.append(
                ("agent_termination_failure", record_host_startup)
            )
        if agent.inference is not None:
            startup_cleanup_steps.append(
                (
                    "inference_gateway_termination_failure",
                    lambda: reference_runtime.stop_container(gateway_name),
                )
            )
        startup_cleanup_steps.extend(
            (
                "runtime_network_teardown_failure",
                lambda network=network: reference_runtime.remove_network(network),
            )
            for network in reversed(networks_to_cleanup)
        )
        startup_cleanup_failures = _run_cleanup_steps(startup_cleanup_steps)
        control_cleanup_failures: List[CleanupFailure] = []
        if supervisor.state["agent_system"]["termination_reason"] is not None:
            control_cleanup_failures = _run_cleanup_steps(
                [
                    (
                        "agent_control_teardown_failure",
                        supervisor.cleanup_agent_control,
                    )
                ]
            )
        reporting_error: Optional[BaseException] = None
        try:
            _report_cleanup_failures(
                supervisor,
                [
                    *startup_cleanup_failures,
                    *control_cleanup_failures,
                ],
            )
        except BaseException as report_exc:
            reporting_error = report_exc
        if transition_error is not None:
            raise transition_error from startup_error
        if reporting_error is not None:
            raise reporting_error from startup_error
        if startup_error is exc:
            raise
        raise startup_error from exc

    terminal_at: Optional[float] = None
    host_termination_reason: Optional[str] = None
    try:
        while True:
            processed = supervisor.process_one_request()
            supervisor.enforce_deadlines()
            status = supervisor.public_status()["status"]
            agent_running = (
                reference_runtime.container_running(agent_name)
                if agent.runner == "container"
                else bool(host_agent and host_agent.running)
            )

            if (
                status == "active"
                and agent.inference is not None
                and not reference_runtime.container_running(gateway_name)
            ):
                supervisor.infrastructure_failed("inference_gateway_exited")
                status = supervisor.public_status()["status"]

            if status == "active" and not agent_running:
                supervisor.agent_exited_without_finish()
                status = "dnf"

            if status != "active":
                if gateway_started:
                    reference_runtime.stop_container(gateway_name)
                    gateway_started = False
                    supervisor.refresh_inference_usage()
                if terminal_at is None:
                    terminal_at = time.monotonic()
                terminal_action = _terminal_agent_action(
                    runner=agent.runner,
                    running=agent_running,
                    drain_elapsed_seconds=time.monotonic() - terminal_at,
                )
                if terminal_action != "wait":
                    if agent.runner == "host_cli":
                        host_termination_reason = terminal_action
                    break
            if not processed:
                time.sleep(0.05)
    except KeyboardInterrupt:
        supervisor.infrastructure_failed("supervisor_interrupted")
        host_termination_reason = "supervisor_interrupted"
    except Exception as exc:
        supervisor.infrastructure_failed(
            "benchmark_monitor_failure", private_detail=str(exc)
        )
        raise
    finally:
        cleanup_failures: List[CleanupFailure] = []
        cleanup_failures.extend(
            _run_cleanup_steps(
                [
                    (
                        "evaluation_worker_teardown_failure",
                        reference_runtime.teardown_pending_evaluations,
                    )
                ]
            )
        )
        active = supervisor.state.get("active_lease")
        pending_lease_teardowns = supervisor.state.get("pending_lease_teardowns", [])
        if active or pending_lease_teardowns:
            cleanup_failures.extend(
                _run_cleanup_steps(
                    [
                        (
                            "research_lease_cleanup_failure",
                            supervisor.terminate_worker_leases_for_teardown,
                        )
                    ]
                )
            )
        try:
            if agent.runner == "container":
                agent_log = (
                    reference_runtime.container_logs(agent_name)
                    if agent_started
                    else ""
                )
                agent_log_path = episode_dir / "agent.log"
                agent_log_path.write_text(agent_log, encoding="utf-8")
                agent_log_path.chmod(0o600)
        except (OSError, RuntimeFailure, KeyboardInterrupt):
            pass
        if agent.runner == "container":
            if agent_started:
                agent_cleanup_failures = _run_cleanup_steps(
                    [
                        (
                            "agent_termination_failure",
                            lambda: _stop_and_record_container_agent(
                                runtime=reference_runtime,
                                supervisor=supervisor,
                                name=agent_name,
                                drain_seconds=(
                                    max(0.0, time.monotonic() - terminal_at)
                                    if terminal_at is not None
                                    else 0.0
                                ),
                            ),
                        )
                    ]
                )
                cleanup_failures.extend(agent_cleanup_failures)
            else:
                cleanup_failures.extend(
                    _run_cleanup_steps(
                        [
                            (
                                "agent_termination_failure",
                                lambda: reference_runtime.stop_container(agent_name),
                            )
                        ]
                    )
                )
        elif host_agent is not None:
            if host_termination_reason is None:
                host_termination_reason = (
                    "supervisor_shutdown" if host_agent.running else "exited"
                )
            drain_seconds = (
                max(0.0, time.monotonic() - terminal_at)
                if terminal_at is not None
                else 0.0
            )
            host_cleanup_failures = _run_cleanup_steps(
                [
                    (
                        "agent_termination_failure",
                        lambda: _stop_and_record_host_agent(
                            supervisor=supervisor,
                            host_agent=host_agent,
                            termination_reason=host_termination_reason,
                            drain_seconds=drain_seconds,
                        ),
                    )
                ]
            )
            cleanup_failures.extend(host_cleanup_failures)
        if gateway_started:
            cleanup_failures.extend(
                _run_cleanup_steps(
                    [
                        (
                            "inference_gateway_termination_failure",
                            lambda: reference_runtime.stop_container(gateway_name),
                        )
                    ]
                )
            )
        cleanup_failures.extend(
            _run_cleanup_steps(
                [
                    (
                        "inference_usage_collection_failure",
                        supervisor.refresh_inference_usage,
                    ),
                    *(
                        (
                            "runtime_network_teardown_failure",
                            lambda network=network: reference_runtime.remove_network(
                                network
                            ),
                        )
                        for network in reversed(networks_to_cleanup)
                    ),
                ]
            )
        )
        cleanup_reporting_error: Optional[BaseException] = None
        try:
            _report_cleanup_failures(supervisor, cleanup_failures)
        except BaseException as exc:
            # A persistence failure while recording teardown evidence must not
            # skip the remaining independent cleanup steps.  The Supervisor
            # mutates the fail-closed evidence before attempting persistence,
            # so staged success cannot be published below.
            cleanup_reporting_error = exc
        agent_termination_recorded = (
            supervisor.state["agent_system"]["termination_reason"] is not None
        )
        if agent_termination_recorded:
            control_cleanup_failures = _run_cleanup_steps(
                [
                    (
                        "agent_control_teardown_failure",
                        supervisor.cleanup_agent_control,
                    )
                ]
            )
            try:
                _report_cleanup_failures(supervisor, control_cleanup_failures)
            except BaseException as exc:
                if cleanup_reporting_error is None:
                    cleanup_reporting_error = exc
        if cleanup_reporting_error is not None:
            raise cleanup_reporting_error

        # Hidden evaluation is deliberately outside the Agent-facing request
        # loop.  It starts only after the Agent is confirmed stopped, all
        # Research Leases are gone, and the control transport is cleaned.  A
        # Worker deployment may materialize ANALYSIS/TEST at this point; Core
        # neither downloads hidden data nor selects a platform mechanism.
        if supervisor.state["status"] == "closed":
            postmortem_error: Optional[BaseException] = None
            try:
                supervisor.evaluate_postmortem()
            except BaseException as exc:
                postmortem_error = exc
            evaluation_cleanup_failures = _run_cleanup_steps(
                [
                    (
                        "evaluation_worker_teardown_failure",
                        reference_runtime.teardown_pending_evaluations,
                    )
                ]
            )
            try:
                _report_cleanup_failures(supervisor, evaluation_cleanup_failures)
            except BaseException as exc:
                if postmortem_error is None:
                    postmortem_error = exc
            if postmortem_error is not None:
                raise postmortem_error

    _print(supervisor.public_status())
    return _episode_exit_code(supervisor.state)


def command_report(args: argparse.Namespace) -> int:
    episode = args.episode.resolve()
    state_path = episode / "state.json"
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"episode state not found: {state_path}") from exc
    payload = normalize_result_report(payload)
    payload["inference_usage"] = summarize_inference_trace(
        episode / "inference" / "inference.jsonl"
    )
    _print(payload)
    return 0


def command_challenge_info(args: argparse.Namespace) -> int:
    release = ChallengeRelease.load(args.challenge)
    _print(release.public_manifest())
    return 0


def command_agent(args: argparse.Namespace) -> int:
    client = Client()
    if args.agent_command == "status":
        result = client.status()
    elif args.agent_command == "lease-start":
        result = client.start_lease(args.seconds)
    elif args.agent_command == "lease-exec":
        argv = list(args.argv)
        if argv and argv[0] == "--":
            argv = argv[1:]
        result = client.exec(argv, timeout_seconds=args.timeout)
    elif args.agent_command == "lease-logs":
        result = client.logs(args.limit)
    elif args.agent_command == "lease-commit":
        result = client.commit(args.candidate)
    elif args.agent_command == "declare-incumbent":
        result = client.declare_incumbent(args.candidate_id)
    elif args.agent_command == "lease-stop":
        result = client.stop_lease()
    elif args.agent_command == "finish":
        result = client.finish()
    else:  # pragma: no cover - argparse prevents this
        raise ClientError(f"unknown agent command: {args.agent_command}")
    _print(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autotrain-bench",
        description="AutoTrain-Bench local reference prototype",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="validate and prepare a task")
    prepare.add_argument("--task", required=True, type=Path)
    prepare.add_argument(
        "--no-pull", action="store_true", help="fail instead of pulling missing images"
    )
    prepare.set_defaults(function=command_prepare)

    start = subparsers.add_parser("start", help="start and supervise an episode")
    start.add_argument(
        "--task",
        type=Path,
        help="task manifest; inferred from --challenge in profile-bound runs",
    )
    start.add_argument("--track", choices=["A", "B", "a", "b"])
    start.add_argument(
        "--challenge",
        type=Path,
        help="challenge release manifest defining supported comparison lanes",
    )
    start.add_argument(
        "--track-config",
        help="named track budget from the selected challenge release",
    )
    start.add_argument(
        "--runtime-profile",
        help="named episode-time hardware profile from the selected challenge release",
    )
    episode_destination = start.add_mutually_exclusive_group(required=True)
    episode_destination.add_argument(
        "--episode-dir",
        type=Path,
        help=(
            "exact new episode directory in a dedicated trusted working root "
            "outside the Core/task checkout"
        ),
    )
    start.add_argument(
        "--agent-runner",
        required=True,
        choices=["container", "host_cli"],
        help="how the participant-owned Agent process is launched",
    )
    start.add_argument(
        "--agent-image",
        help="participant Agent image (required for the container runner)",
    )
    start.add_argument(
        "--agent-command",
        required=True,
        help="participant Agent command",
    )
    start.add_argument(
        # Removed in favour of --agent-egress; accepted only so an old script
        # fails with a migration message instead of an argparse usage error.
        "--agent-mode",
        help=argparse.SUPPRESS,
    )
    start.add_argument(
        "--agent-egress",
        choices=["denied", "unrestricted-dev"],
        help=(
            "Agent egress policy; denied gives the Agent no route to the public "
            "internet, while unrestricted-dev is convenient for debugging but "
            "is not policy-compliant"
        ),
    )
    start.add_argument(
        "--agent-env",
        dest="agent_environment",
        action="append",
        default=[],
        metavar="NAME",
        help="forward a named host environment variable to the selected Agent runner",
    )
    start.add_argument(
        "--agent-workspace",
        type=Path,
        help=(
            "snapshot participant-owned initial files into the episode; omit for "
            "an empty workspace"
        ),
    )
    start.add_argument("--inference-base-url")
    start.add_argument("--inference-api-key-env")
    start.add_argument(
        "--inference-model", dest="inference_models", action="append", default=[]
    )
    start.add_argument(
        "--inference-path", dest="inference_paths", action="append", default=[]
    )
    start.add_argument("--inference-timeout", type=int, default=300)
    start.add_argument(
        "--training-gpus",
        type=int,
        default=0,
        metavar="COUNT",
        help=(
            "expose COUNT NVIDIA GPUs to Training Containers and, when the "
            "task requests CUDA evaluation, Candidate Containers"
        ),
    )
    start.add_argument(
        "--training-gpu-model",
        help="require the trusted host GPU inventory to contain this model string",
    )
    start.add_argument(
        "--training-device",
        dest="training_devices",
        action="append",
        default=[],
        metavar="/DEV/PATH",
        help="expose an additional Docker device to Training Containers",
    )
    episode_destination.add_argument(
        "--episodes-root",
        type=Path,
        help=(
            "dedicated trusted root under which a new episode directory is created; "
            "must be outside the Core/task checkout"
        ),
    )
    start.set_defaults(function=command_start)

    report = subparsers.add_parser("report", help="print a completed episode report")
    report.add_argument("--episode", required=True, type=Path)
    report.set_defaults(function=command_report)

    challenge_info = subparsers.add_parser(
        "challenge-info", help="show supported track and runtime profile maps"
    )
    challenge_info.add_argument("--challenge", required=True, type=Path)
    challenge_info.set_defaults(function=command_challenge_info)

    agent = subparsers.add_parser(
        "agent", help="mailbox commands used by the selected Agent runner"
    )
    agent_subparsers = agent.add_subparsers(dest="agent_command", required=True)

    agent_status = agent_subparsers.add_parser("status")
    agent_status.set_defaults(function=command_agent)

    lease_start = agent_subparsers.add_parser("lease-start")
    lease_start.add_argument("--seconds", type=int)
    lease_start.set_defaults(function=command_agent)

    lease_exec = agent_subparsers.add_parser("lease-exec")
    lease_exec.add_argument("--timeout", type=float)
    lease_exec.add_argument("argv", nargs=argparse.REMAINDER)
    lease_exec.set_defaults(function=command_agent)

    lease_logs = agent_subparsers.add_parser("lease-logs")
    lease_logs.add_argument("--limit", type=int, default=50)
    lease_logs.set_defaults(function=command_agent)

    lease_commit = agent_subparsers.add_parser("lease-commit")
    lease_commit.add_argument(
        "candidate",
        metavar="/output/CANDIDATE",
        help="lease-owned Candidate directory produced inside the Training Container",
    )
    lease_commit.set_defaults(function=command_agent)

    declare_incumbent = agent_subparsers.add_parser("declare-incumbent")
    declare_incumbent.add_argument("candidate_id")
    declare_incumbent.set_defaults(function=command_agent)

    lease_stop = agent_subparsers.add_parser("lease-stop")
    lease_stop.set_defaults(function=command_agent)

    finish = agent_subparsers.add_parser("finish")
    finish.set_defaults(function=command_agent)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "track", None):
        args.track = args.track.upper()
    try:
        return_code = args.function(args)
    except (ClientError, ConfigError, RuntimeFailure, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return_code = 2
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
