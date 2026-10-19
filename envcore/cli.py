"""Command-line interface for users and containerized agents."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import sys
import time
import uuid
from contextlib import ExitStack
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .agent_api_bundle import publish_agent_api_bundle
from .benchmark import BenchmarkRelease, ResolvedBenchmarkRun
from .environment import EnvironmentConfig
from .run_config import EnvironmentRun, resolve_environment_run
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
    resolve_operator_research_limit,
    resolve_operator_storage_limit,
)
from .host_agent import (
    HostAgentProcess,
    HostAgentStartupFailure,
)
from .docker_recovery import DEFAULT_RECOVERY_TIMEOUT_SECONDS, recovery_timeout_seconds
from .inference import summarize_inference_trace
from .local_mailbox import LocalFileMailbox
from .local_agent_execution import describe_local_agent_execution
from .local_run_guard import LocalResearchGuard
from .local_watchdog import (
    CONTROL_TIMEOUT,
    DEADLINE_GRACE_SECONDS,
    HEARTBEAT_SECONDS,
    HEARTBEAT_TIMEOUT,
    current as current_watchdog,
    supervise_start,
)
from .results import normalize_result_report
from .operator_status import (
    DEFAULT_MAX_AGE_SECONDS,
    format_operator_status,
    read_operator_status,
)
from .runtime import DockerRuntime, RuntimeFailure
from .supervisor import Supervisor
from .training_data import verify_training_data


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
    operator_stopped: bool = False,
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
        termination_reason=(
            "operator_time_limit_reached" if operator_stopped else termination_reason
        ),
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
    usage = host_agent.summarize_usage()
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
    training_data_verification = verify_training_data(config.train_dir)
    reference_runtime = DockerRuntime()
    reference_runtime.prepare(config, pull_missing=not args.no_pull)
    worker_environment = reference_runtime.environment_manifest(config)
    _print(
        {
            "status": "ready",
            "training_data_verification": training_data_verification,
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


def _resolve_start_run(
    args: argparse.Namespace,
) -> Tuple[EnvironmentRun, Optional[ResolvedBenchmarkRun]]:
    """Select input presets at the application boundary, then use one lifecycle."""
    release = BenchmarkRelease.load(args.benchmark) if args.benchmark else None
    if release is None:
        if args.task is None:
            raise ConfigError("--task is required when --benchmark is not used")
        if args.track_config or args.runtime_profile:
            raise ConfigError(
                "--track-config and --runtime-profile require --benchmark"
            )
        if not args.track:
            raise ConfigError("--track is required when --benchmark is not used")
        task_path = args.task
    else:
        task_path = release.task_path
        if args.task is not None and args.task.expanduser().resolve() != task_path:
            raise ConfigError("--task may not override the task owned by --benchmark")
        if args.track or not args.track_config:
            raise ConfigError("--benchmark requires --track-config instead of --track")
    config = TaskConfig.load(task_path)
    manual = args.training_gpus or args.training_gpu_model or args.training_devices
    if manual and (args.environment or release is not None):
        raise ConfigError(
            "manual training resources may not override a selected environment; use --environment"
        )
    environment = EnvironmentConfig.load(args.environment) if args.environment else None
    if release is not None:
        run = release.resolve(
            config,
            track_config_id=args.track_config,
            runtime_profile_id=args.runtime_profile,
            environment=environment,
        )
        return run, run
    run = resolve_environment_run(
        replace(config, leaderboard_eligible=False),
        track=args.track,
        environment=environment,
        resources=None
        if environment is not None
        else TrainingResources(
            nvidia_gpu_count=args.training_gpus,
            required_gpu_model=args.training_gpu_model,
            devices=args.training_devices,
        ),
    )
    return run, None


def command_start(args: argparse.Namespace) -> int:
    with ExitStack() as lifetime:
        return _command_start(args, lifetime)


def _command_start(args: argparse.Namespace, lifetime: ExitStack) -> int:
    if getattr(args, "agent_mode", None) is not None:
        raise ConfigError(
            "--agent-mode was renamed to --agent-egress and its values changed: "
            "'sandboxed' is now 'denied'; 'unrestricted-dev' is unchanged"
        )
    run, benchmark = _resolve_start_run(args)
    config, track = run.task, run.track
    resources = run.resources
    resource_envelope = run.resource_envelope
    evaluation_resource_envelope = run.evaluation_resource_envelope
    run_environment = run.environment
    comparison_key = (
        benchmark.comparison_key.public_manifest()
        if benchmark is not None and benchmark.comparison_key is not None
        else None
    )
    benchmark_release = benchmark.release_manifest if benchmark is not None else None
    operator_limit = resolve_operator_research_limit(
        config.source,
        track,
        override=args.operator_max_research_seconds,
        disabled=args.no_operator_research_limit,
    )
    storage_limit = resolve_operator_storage_limit(
        config.source,
        override=args.operator_max_storage_bytes,
        disabled=args.no_operator_storage_limit,
    )
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
    if track == "B":
        if operator_limit is None:
            print(
                "WARNING: Track B operator research-time protection is disabled. "
                "Research may continue indefinitely and token consumption may be "
                "extremely high. Set --operator-max-research-seconds SECONDS to "
                "enable protection."
            )
        else:
            print(
                f"Track B operator research-time protection: {operator_limit:g} seconds "
                "from task reveal. Override with --operator-max-research-seconds "
                "SECONDS or disable with --no-operator-research-limit. Disabling "
                "it risks extremely high token consumption."
            )
        print(
            "This is a private operator setting; it is not sent to the Tested Agent.",
            flush=True,
        )
    if storage_limit is None:
        print(
            "WARNING: Operator storage protection is disabled; writable episode "
            "storage may exhaust the Worker disk. "
            "Enable it before launch with --operator-max-storage-bytes BYTES.",
            flush=True,
        )
    else:
        storage_origin = (
            "task default"
            if args.operator_max_storage_bytes is None
            else "run override"
        )
        print(
            f"Operator storage protection: {storage_limit / 2**30:g} GiB "
            f"({storage_limit} bytes; {storage_origin}) for this episode, on either Track.\n"
            "Change it before launch with --operator-max-storage-bytes BYTES, "
            "or disable it with --no-operator-storage-limit. "
            "WARNING: disabling protection risks Worker disk exhaustion.\n"
            "The Worker requires a dedicated bounded filesystem at the episode "
            "parent directory.",
            flush=True,
        )
    print(
        "This is a private operator setting; it is not sent to the Tested Agent.",
        flush=True,
    )
    if storage_limit is not None and resolved_runner == "host_cli":
        raise ConfigError(
            "host_cli cannot confine Agent writes to protected storage; use the "
            "container runner or explicitly pass --no-operator-storage-limit "
            "for an unprotected preview"
        )
    # Admission belongs to the execution-side Worker, before workspace copying
    # or task reveal. This local composition needs a shared Docker filesystem;
    # remote deployments perform equivalent admission in their own environment.
    training_data_verification = verify_training_data(config.train_dir)
    reference_runtime = DockerRuntime(
        hidden_datasets={"analysis": config.analysis_path, "test": config.test_path}
    )
    reference_runtime.preflight_training_resources(resource_envelope)
    reference_runtime.preflight_evaluation_resources(
        config,
        resources=resources,
        resource_envelope=evaluation_resource_envelope,
    )
    if storage_limit is not None:
        reference_runtime.configure_operator_storage(
            root=episode_dir.parent,
            max_bytes=storage_limit,
            probe_image=config.evaluator_image,
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
        agent_execution = describe_local_agent_execution(
            agent, operator_storage_bounded=storage_limit is not None
        )
        missing_environment = [
            name for name in agent.environment if name not in os.environ
        ]
        if missing_environment:
            names = ", ".join(missing_environment)
            raise ConfigError(f"agent environment variable is not set: {names}")
        # Local file locations are private composition inputs for the bundled
        # Worker.  They never cross the semantic Core/Worker protocol, and may
        # remain absent until post-research evaluation begins.
        reference_runtime.prepare(
            config,
            agent=agent,
            resources=resources,
            pull_missing=False,
        )
        if resolved_runner == "container":
            state_dir = args.worker_state_dir or (
                Path.home()
                / ".envcore"
                / "workers"
                / uuid.uuid5(uuid.NAMESPACE_URL, str(episode_dir)).hex
            )
            state_dir = state_dir.expanduser().resolve()
            if state_dir == episode_dir or episode_dir in state_dir.parents:
                raise ConfigError(
                    "Worker state must be outside the episode storage domain"
                )
            if storage_limit is not None and (
                state_dir == episode_dir.parent
                or episode_dir.parent in state_dir.parents
            ):
                raise ConfigError(
                    "Worker state must be outside protected episode storage"
                )
            # Readiness probes above are complete. Every episode workload and
            # post-research hidden materialization below is now recoverable.
            reference_runtime.enable_recovery(
                state_dir=state_dir, episode_dir=episode_dir
            )
            lifetime.callback(reference_runtime.close_recovery)
            print(
                f"Worker cleanup journal: {state_dir}\n"
                "After Worker process loss, run worker-recover --state-dir PATH "
                "in this same Worker environment. Recovery does not resume the episode.",
                flush=True,
            )
            if current_watchdog() is not None:
                print(
                    "Independent Worker watchdog enabled: process loss or a missed "
                    "deadline triggers termination and journaled cleanup. "
                    f"Heartbeat timeout {args.worker_watchdog_heartbeat_seconds:g}s; "
                    f"deadline grace {args.worker_watchdog_grace_seconds:g}s; "
                    f"control-call fallback {args.worker_watchdog_control_seconds:g}s; "
                    f"recovery allowance {args.worker_watchdog_recovery_seconds:g}s. "
                    "Adjust these before launch with --worker-watchdog-heartbeat-seconds, "
                    "--worker-watchdog-grace-seconds, --worker-watchdog-control-seconds "
                    "and --worker-watchdog-recovery-seconds. "
                    "These are private operator settings; recovery does not resume the episode.",
                    flush=True,
                )
        elif args.worker_state_dir is not None:
            raise ConfigError("Worker crash recovery is not supported for host_cli")
        worker_environment = reference_runtime.environment_manifest(
            config,
            agent=agent,
            resources=resources,
        )
        supervisor = Supervisor(
            config=config,
            agent=agent,
            resources=resources,
            training_data_verification=training_data_verification,
            resource_envelope=resource_envelope,
            evaluation_resource_envelope=evaluation_resource_envelope,
            track=track,
            episode_dir=episode_dir,
            agent_api_dir=agent_api_dir,
            agent_execution=agent_execution,
            research_worker=reference_runtime,
            evaluation_worker=reference_runtime,
            agent_control_transport=agent_control,
            research_worker_environment=worker_environment,
            evaluation_worker_environment=worker_environment,
            operator_max_research_seconds=operator_limit,
            comparison_key=comparison_key,
            benchmark_release=benchmark_release,
            environment=run_environment,
        )
    except Exception:
        # The public helper bundle is read-only. No Agent has started yet;
        # restore its directory permissions so a failed preflight can remove it.
        bundle = episode_dir / "public" / "agent_api"
        for directory in (bundle / "envcore", bundle):
            if directory.is_dir():
                directory.chmod(0o755)
        shutil.rmtree(episode_dir, ignore_errors=True)
        raise
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

    def stop_operator_research() -> List[CleanupFailure]:
        # Use a separate local client and persisted semantic Lease IDs. Never
        # wait for the Supervisor lock held by a synchronous Worker operation.
        guard_runtime = DockerRuntime(docker_binary=reference_runtime.docker)
        reference_runtime.share_recovery_with(guard_runtime)
        steps: List[CleanupStep] = []
        if agent.inference is not None:
            steps.append(
                (
                    "inference_gateway_termination_failure",
                    lambda: guard_runtime.stop_container(gateway_name),
                )
            )
        if agent.runner == "container":
            steps.append(
                (
                    "agent_termination_failure",
                    lambda: guard_runtime.stop_container(agent_name),
                )
            )
        elif host_agent is not None:
            steps.append(("agent_termination_failure", host_agent.stop))
        failures = _run_cleanup_steps(steps)
        try:
            state = json.loads(supervisor.state_path.read_text(encoding="utf-8"))
            lease_ids = set(state.get("pending_lease_teardowns", []))
            if state.get("active_lease"):
                lease_ids.add(state["active_lease"]["lease_id"])
        except Exception as exc:
            failures.append(("operator_guard_state_failure", str(exc)))
            return failures
        failures.extend(
            _run_cleanup_steps(
                [
                    (
                        "research_lease_cleanup_failure",
                        lambda lease_id=lease_id: guard_runtime.terminate_lease(
                            episode_id=state["episode_id"], lease_id=lease_id
                        ),
                    )
                    for lease_id in sorted(lease_ids)
                ]
            )
        )
        return failures

    operator_guard = (
        LocalResearchGuard(supervisor.operator_limit_remaining, stop_operator_research)
        if operator_limit is not None
        else None
    )
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
        watchdog = current_watchdog()
        if watchdog is not None and operator_limit is not None:
            watchdog.arm(
                "research",
                supervisor.operator_limit_remaining(),
                "operator_research_limit",
            )
        if operator_guard is not None:
            operator_guard.start()
        if agent.runner == "container":
            agent_start_options = {}
            if operator_guard is not None:
                agent_start_options["timeout_seconds"] = (
                    supervisor.operator_limit_remaining()
                )
            reference_runtime.start_agent(
                name=agent_name,
                config=config,
                agent=agent,
                control_dir=agent_control.control_dir,
                public_task_path=supervisor.public_task_path,
                inference_network=agent_network
                if agent.inference is not None
                else None,
                **agent_start_options,
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
        transition_error: Optional[BaseException] = None
        try:
            supervisor.enforce_operator_limit()
        except BaseException as transition_exc:
            transition_error = transition_exc
        guard_failures = operator_guard.close() if operator_guard is not None else []
        if isinstance(exc, HostAgentStartupFailure):
            # Popen succeeded, so ``None`` would be an unsafe statement about
            # process ownership.  Preserve the retryable handle even when the
            # launch helper's first termination attempt also failed.
            host_agent = exc.handle
            startup_error = exc.startup_error
        termination_reason = (
            "operator_time_limit_reached"
            if supervisor.state["status"] == "operator_stopped"
            else "supervisor_interrupted"
            if isinstance(startup_error, KeyboardInterrupt)
            else "episode_startup_failure"
        )
        private_detail = f"{type(startup_error).__name__}: {startup_error}"
        if isinstance(exc, HostAgentStartupFailure) and exc.cleanup_error is not None:
            private_detail += (
                "; initial host Agent cleanup failed: "
                f"{type(exc.cleanup_error).__name__}: {exc.cleanup_error}"
            )

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
                    *guard_failures,
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
            supervisor.enforce_operator_limit()
            if supervisor.state["status"] == "operator_stopped":
                host_termination_reason = "operator_time_limit_reached"
                break
            processed = supervisor.process_one_request()
            supervisor.enforce_deadlines()
            status = supervisor.public_status()["status"]
            if (
                status not in {"active", "operator_stopped"}
                and current_watchdog() is not None
            ):
                current_watchdog().disarm("research")
            if status == "operator_stopped":
                host_termination_reason = "operator_time_limit_reached"
                break
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
        operator_recording_error: Optional[BaseException] = None
        try:
            supervisor.enforce_operator_limit()
        except BaseException as exc:
            # A failed audit write must not prevent workload revocation.
            operator_recording_error = exc
        cleanup_failures: List[CleanupFailure] = (
            operator_guard.close() if operator_guard is not None else []
        )
        if supervisor.state["status"] == "operator_stopped":
            # Stop inference/Agent first, even if Lease cleanup will fail. No
            # normal finish drain is granted after an operator protection trip.
            host_termination_reason = "operator_time_limit_reached"
            cleanup_failures.extend(stop_operator_research())
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
                    reference_runtime.container_logs(agent_name, timeout_seconds=5)
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
                                operator_stopped=(
                                    supervisor.state["status"] == "operator_stopped"
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
        cleanup_reporting_error: Optional[BaseException] = operator_recording_error
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
        postmortem_error: Optional[BaseException] = None
        if supervisor.state["status"] == "closed":
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
        private_materialization_failures = _run_cleanup_steps(
            [
                (
                    "evaluation_private_storage_teardown_failure",
                    reference_runtime.close_private_materializations,
                )
            ]
        )
        try:
            _report_cleanup_failures(supervisor, private_materialization_failures)
        except BaseException as exc:
            if postmortem_error is None:
                postmortem_error = exc
        if postmortem_error is not None:
            raise postmortem_error
        if supervisor.state["status"] == "closed":
            supervisor.publish_postmortem_result()

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


def command_worker_recover(args: argparse.Namespace) -> int:
    timeout_seconds = recovery_timeout_seconds(args.timeout_seconds)
    report = DockerRuntime().recover_resources(
        args.state_dir, timeout_seconds=timeout_seconds
    )
    _print(report)
    return (
        0
        if report["cleanup_complete"]
        and not report.get("process_supervision", {}).get("operator_signal")
        else 2
    )


def command_status(args: argparse.Namespace) -> int:
    view = read_operator_status(
        args.episode,
        observation_path=args.observation,
        max_age_seconds=args.max_age_seconds,
    )
    if args.json:
        _print(view)
    else:
        print(format_operator_status(view))
    return 0


def command_benchmark_info(args: argparse.Namespace) -> int:
    release = BenchmarkRelease.load(args.benchmark)
    _print(release.public_manifest())
    return 0


def command_environment_info(args: argparse.Namespace) -> int:
    """Inspect a requested configuration without data, Docker or allocation."""
    config = EnvironmentConfig.load(args.environment)
    _print(config.record())
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
        prog="envcore",
        description="Configurable autonomous research environments and benchmark protocols",
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
        "--environment",
        type=Path,
        help="run-owned compute resource and time-budget JSON; freezes settings before task reveal",
    )
    start.add_argument(
        "--task",
        type=Path,
        help="task manifest; inferred from --benchmark in profile-bound runs",
    )
    start.add_argument("--track", choices=["A", "B", "a", "b"])
    start.add_argument(
        "--benchmark",
        type=Path,
        help="benchmark release manifest defining supported comparison lanes",
    )
    start.add_argument(
        "--track-config",
        help="named track budget from the selected benchmark release",
    )
    start.add_argument(
        "--runtime-profile",
        help="named episode-time hardware profile from the selected benchmark release",
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
    operator_protection = start.add_mutually_exclusive_group()
    operator_protection.add_argument(
        "--operator-max-research-seconds",
        type=float,
        metavar="SECONDS",
        help=(
            "override the task's private Track B research-time protection in seconds; "
            "defaults: lm-pretrain 86400 (24h), smoke_lm-pretrain 1800 (30m); "
            "best-effort in the local launcher, never an Agent budget"
        ),
    )
    operator_protection.add_argument(
        "--no-operator-research-limit",
        action="store_true",
        help=(
            "disable private Track B research-time protection for this run; "
            "WARNING: token consumption may be extremely high"
        ),
    )
    storage_protection = start.add_mutually_exclusive_group()
    storage_protection.add_argument(
        "--operator-max-storage-bytes",
        type=int,
        metavar="BYTES",
        help=(
            "private episode storage protection on either Track; requires a "
            "deployment-provided bounded filesystem at the episode parent; "
            "defaults: lm-pretrain 100 GiB, smoke_lm-pretrain 2 GiB"
        ),
    )
    storage_protection.add_argument(
        "--no-operator-storage-limit",
        action="store_true",
        help="disable private episode storage protection; WARNING: disk exhaustion risk",
    )
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
    start.add_argument(
        "--worker-state-dir",
        type=Path,
        help="private local Worker recovery directory outside episode storage; "
        "defaults to ~/.envcore/workers/<episode-path-id>",
    )
    for name, default, help_text in (
        ("heartbeat", HEARTBEAT_TIMEOUT, "missing Worker heartbeat tolerance"),
        ("grace", DEADLINE_GRACE_SECONDS, "grace for cooperative deadline handling"),
        (
            "control",
            CONTROL_TIMEOUT,
            "fallback for control/startup calls without a timeout",
        ),
        ("recovery", DEFAULT_RECOVERY_TIMEOUT_SECONDS, "shared recovery allowance"),
    ):
        start.add_argument(
            f"--worker-watchdog-{name}-seconds",
            type=float,
            default=default,
            metavar="SECONDS",
            help=f"private Docker watchdog {help_text} (default: {default:g})",
        )

    recover = subparsers.add_parser(
        "worker-recover",
        help="reclaim a stopped Docker Worker's resources without resuming",
    )
    recover.add_argument("--state-dir", type=Path, required=True)
    recover.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_RECOVERY_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=f"shared recovery allowance (default: {DEFAULT_RECOVERY_TIMEOUT_SECONDS:g})",
    )
    recover.set_defaults(function=command_worker_recover)

    status = subparsers.add_parser(
        "status", help="show trusted Operator execution evidence and Core snapshot"
    )
    status.add_argument("--episode", required=True, type=Path)
    status.add_argument(
        "--observation", type=Path, help="trusted deployment observation JSON"
    )
    status.add_argument(
        "--max-age-seconds",
        type=float,
        default=DEFAULT_MAX_AGE_SECONDS,
        help="maximum deployment observation age (default: 60 seconds)",
    )
    status.add_argument(
        "--json", action="store_true", help="emit the structured Operator view"
    )
    status.set_defaults(function=command_status)

    report = subparsers.add_parser(
        "report", help="print the full trusted Core episode report"
    )
    report.add_argument("--episode", required=True, type=Path)
    report.set_defaults(function=command_report)

    benchmark_info = subparsers.add_parser(
        "benchmark-info", help="show supported track and runtime profile maps"
    )
    benchmark_info.add_argument("--benchmark", required=True, type=Path)
    benchmark_info.set_defaults(function=command_benchmark_info)
    environment_info = subparsers.add_parser(
        "environment-info", help="validate and fingerprint an environment configuration"
    )
    environment_info.add_argument("--environment", required=True, type=Path)
    environment_info.set_defaults(function=command_environment_info)

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
        if args.function is command_start and args.agent_runner == "container":
            for value in (
                args.worker_watchdog_control_seconds,
                args.worker_watchdog_grace_seconds,
                args.worker_watchdog_heartbeat_seconds,
                args.worker_watchdog_recovery_seconds,
            ):
                if not math.isfinite(value) or value <= 0:
                    raise ConfigError(
                        "Worker watchdog settings must be finite positive seconds"
                    )
            if args.worker_watchdog_heartbeat_seconds <= HEARTBEAT_SECONDS:
                raise ConfigError(
                    "Worker watchdog heartbeat tolerance must exceed 0.5 seconds"
                )
            if current_watchdog() is not None:
                current_watchdog().control_timeout = (
                    args.worker_watchdog_control_seconds
                )
        if (
            args.function is command_start
            and args.agent_runner == "container"
            and current_watchdog() is None
        ):
            return_code = supervise_start(
                list(argv) if argv is not None else sys.argv[1:],
                heartbeat_timeout=args.worker_watchdog_heartbeat_seconds,
                grace_seconds=args.worker_watchdog_grace_seconds,
                recovery_seconds=args.worker_watchdog_recovery_seconds,
            )
        else:
            return_code = args.function(args)
    except (ClientError, ConfigError, RuntimeFailure, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return_code = 2
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
