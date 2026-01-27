from netket.utils.timing import Timer as TimerNK
from netket.utils.timing import _NULLTIMER, CURRENT_TIMER_STACK, display
import contextlib
import inspect
from rich.tree import Tree
from rich.panel import Panel

@display.rich_repr
class Timer(TimerNK):
    total: float
    sub_timers: dict
    _start_time: float = 0.0

    # List of subtimers to exclude from percentage calculation
    _exclude_from_stats: set[str]

    def __init__(self, exclude_from_stats=None):
        super().__init__()
        self._exclude_from_stats = set(exclude_from_stats) if exclude_from_stats else set()

    def _get_excluded_total(self):
        """Recursively calculate total time of excluded timers."""
        excluded = 0.0
        for k, sub in self.sub_timers.items():
            if k in self._exclude_from_stats:
                # This entire subtimer is excluded
                excluded += sub.total
            else:
                # Check if this subtimer has excluded children
                excluded += sub._get_excluded_total()
        return excluded

    def _get_relevant_total(self):
        """Calculate total time excluding timers in _exclude_from_stats."""
        return self.total - self._get_excluded_total()

    def __rich__(self, indent=0, total_above=None):
        """Override the rich display to show relevant total in the tree."""
        relevant_total = self._get_relevant_total()
        excluded_total = self._get_excluded_total()
        
        # Build the title with relevant total
        if excluded_total > 0:
            tree = Tree(f"Total (relevant): {relevant_total:.3f} | Total (excluded): {excluded_total:.3f} | Total (all): {self.total:.3f}")
        else:
            tree = Tree(f"Total: {self.total:.3f}")
            
        self._rich_walk_tree_(tree)
        return Panel(tree, title="Timing Information")

    def _rich_walk_tree_(self, tree):
        # Use relevant total for percentage calculations at this level
        relevant_total = sum(
            sub.total for k, sub in self.sub_timers.items()
            if k not in self._exclude_from_stats
        )
        
        attributed = 0.0
        for key, sub_timer in self.sub_timers.items():
            if key in self._exclude_from_stats:
                # Flag excluded timers clearly - always show them
                sub_tree = tree.add(f"[dim]⊗ {key} : {sub_timer.total:.3f} s [EXCLUDED][/dim]")
                # Still walk inside to show the subtree structure
                sub_timer._rich_walk_tree_(sub_tree)
                continue

            # Only show timers that are > 1% of relevant total
            if relevant_total > 0 and sub_timer.total / relevant_total > 0.01:
                percentage = 100 * (sub_timer.total / relevant_total)
                attributed += sub_timer.total

                sub_tree = tree.add(
                    f"({percentage:.1f}%) | {key} : {sub_timer.total:.3f} s"
                )
                sub_timer._rich_walk_tree_(sub_tree)

    def get_subtimer(self, name: str):
        if name not in self.sub_timers:
            # Pass the exclude list to child timers
            self.sub_timers[name] = Timer(exclude_from_stats=self._exclude_from_stats)
        return self.sub_timers[name]
    
@contextlib.contextmanager
def timed_scope(name: str = None, force: bool = False, exclude_from_stats: bool = False):
    """
    Context manager to time a code block.

    Args:
        name: Name of the timer. If None, it will be inferred from the caller.
        force: If True, force timing even if no parent timer exists.
        exclude_from_stats: If True, this timer and its subtree are excluded from percentages.
    """
    global CURRENT_TIMER_STACK

    if name is None:
        caller_frame = inspect.stack()[2]
        frame = caller_frame[0]
        info = inspect.getframeinfo(frame)
        name = f"{info.filename}:{info.lineno}"

    if force or len(CURRENT_TIMER_STACK) > 0:
        if len(CURRENT_TIMER_STACK) == 0:
            timer = Timer()
            if exclude_from_stats:
                timer._exclude_from_stats.add(name)
        else:
            parent_timer = CURRENT_TIMER_STACK[-1]
            # Mark this timer as excluded in the parent
            if exclude_from_stats:
                parent_timer._exclude_from_stats.add(name)
            timer = parent_timer.get_subtimer(name)

        CURRENT_TIMER_STACK.append(timer)
        try:
            with timer:
                yield timer
        finally:
            CURRENT_TIMER_STACK.pop()
    else:
        yield _NULLTIMER