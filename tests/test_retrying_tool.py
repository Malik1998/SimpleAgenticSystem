from simple_agentic_system.tools import RepairPolicy, RetryingTool, ToolResult


class FlakyTool:
    name = "flaky"
    description = "fails N times then succeeds"
    parameters: dict = {}

    def __init__(self, fail_times: int = 1):
        self.calls = 0
        self.fail_times = fail_times

    async def run(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            return ToolResult(is_error=True, error="boom")
        return ToolResult(output=kwargs["x"] * 2)


class IncrementFixer:
    name = "fixer"
    description = "bumps x by 1"
    parameters: dict = {}

    def __init__(self):
        self.calls = 0

    async def run(self, **kwargs):
        self.calls += 1
        original = kwargs["original_arguments"]
        return ToolResult(output={"x": original["x"] + 1})


async def test_success_on_first_try_never_calls_fixer():
    flaky = FlakyTool(fail_times=0)
    fixer = IncrementFixer()
    rt = RetryingTool(inner=flaky, fixer=fixer, policy=RepairPolicy())
    result = await rt.run(x=5)
    assert result.output == 10
    assert fixer.calls == 0


async def test_sequential_repair_succeeds():
    flaky = FlakyTool(fail_times=1)
    rt = RetryingTool(inner=flaky, fixer=IncrementFixer(), policy=RepairPolicy(max_attempts=2, mode="sequential"))
    result = await rt.run(x=5)
    assert not result.is_error
    assert result.output == 12  # (5 + 1) * 2
    assert flaky.calls == 2


async def test_sequential_gives_up_after_max_attempts():
    flaky = FlakyTool(fail_times=10)
    rt = RetryingTool(inner=flaky, fixer=IncrementFixer(), policy=RepairPolicy(max_attempts=2, mode="sequential"))
    result = await rt.run(x=5)
    assert result.is_error


async def test_parallel_repair_succeeds():
    flaky = FlakyTool(fail_times=1)
    rt = RetryingTool(
        inner=flaky,
        fixer=IncrementFixer(),
        policy=RepairPolicy(max_attempts=2, mode="parallel", parallel_fanout=3),
    )
    result = await rt.run(x=5)
    assert not result.is_error
    assert result.output == 12


async def test_retrying_tool_exposes_inner_spec_by_default():
    flaky = FlakyTool()
    rt = RetryingTool(inner=flaky, fixer=IncrementFixer())
    assert rt.name == "flaky"
    assert rt.description == flaky.description
