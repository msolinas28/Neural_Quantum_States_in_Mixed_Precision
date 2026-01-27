from collections.abc import Callable, Iterable
import numbers

from tqdm.auto import tqdm
from netket.logging import AbstractLog, JsonLog
from netket.operator._abstract_observable import AbstractObservable
from netket.utils import mpi, timing
from netket.driver import AbstractVariationalDriver, VMC
from netket.driver.abstract_variational_driver import _to_iterable

CallbackT = Callable[[int, dict, "AbstractVariationalDriver"], bool]

class time_VMC(VMC):
    def run(
        self,
        n_iter: int,
        out: AbstractLog | Iterable[AbstractLog] | str | None = (),
        obs: dict[str, AbstractObservable] | None = None,
        step_size: int = 1,
        show_progress: bool = True,
        save_params_every: int = 50,  # for default logger
        write_every: int = 50,  # for default logger
        callback: CallbackT | Iterable[CallbackT] = lambda *x: True,
        timeit: bool = False,
    ):
        """
        Runs this variational driver, updating the weights of the network stored in
        this driver for `n_iter` steps and dumping values of the observables `obs`
        in the output `logger`.

        It is possible to control more specifically what quantities are logged, when to
        stop the optimisation, or to execute arbitrary code at every step by specifying
        one or more callbacks, which are passed as a list of functions to the keyword
        argument `callback`.

        Callbacks are functions that follow this signature:

        .. Code::

            def callback(step, log_data, driver) -> bool:
                ...
                return True/False

        If a callback returns True, the optimisation continues, otherwise it is stopped.
        The `log_data` is a dictionary that can be modified in-place to change what is
        logged at every step. For example, this can be used to log additional quantities
        such as the acceptance rate of a sampler.

        Loggers are specified as an iterable passed to the keyword argument `out`. If only
        a string is specified, this will create by default a :class:`nk.logging.JsonLog`.
        To know about the output format check its documentation. The logger object is
        also returned at the end of this function so that you can inspect the results
        without reading the json output.

        When running among multiple MPI ranks/Jax devices, the logging logic is executed
        on all nodes, but only root-rank loggers should write to files or do expensive I/O
        operations.

        .. note::

            Before NetKet 3.15, loggers where automatically 'ignored' on non-root ranks.
            However, starting with NetKet 3.15 it is the responsability of a logger to
            check if it is executing on a non-root rank, and to 'do nothing' if that is
            the case.

            The change was required to work correctly and efficiently with sharding. It will
            only affect users that were defining custom loggers themselves.

        Args:
            n_iter: the total number of iterations to be performed during this run.
            out: A logger object, or an iterable of loggers, to be used to store simulation log and data.
                If this argument is a string, it will be used as output prefix for the standard JSON logger.
            obs: An iterable containing all observables that should be computed
            step_size: Every how many steps should observables be logged to disk (default=1)
            callback: Callable or list of callable callback functions to stop training given a condition
            show_progress: If true displays a progress bar (default=True)
            save_params_every: Every how many steps the parameters of the network should be
                serialized to disk (ignored if logger is provided)
            write_every: Every how many steps the json data should be flushed to disk (ignored if
                logger is provided)
            timeit: If True, provide timing information.
        """

        if not isinstance(n_iter, numbers.Number):
            raise ValueError(
                "n_iter, the first positional argument to `run`, must be a number!"
            )

        if obs is None:
            obs = {}

        # if out is a path, create an overwriting Json Log for output
        if isinstance(out, str):
            out = JsonLog(out, "w", save_params_every, write_every)
        elif out is None:
            out = ()

        loggers = _to_iterable(out)
        callbacks = _to_iterable(callback)
        callback_stop = False

        with timing.timed_scope(force=timeit) as timer:
            with tqdm(
                total=n_iter,
                disable=not show_progress or not self._is_root,
                dynamic_ncols=True,
            ) as pbar:
                old_step = self.step_count
                first_step = True

                for step in self.iter(n_iter, step_size):
                    log_data = self.estimate(obs)
                    self._log_additional_data(log_data, step)

                    # if the cost-function is defined then report it in the progress bar
                    if self._loss_stats is not None:
                        pbar.set_postfix_str(
                            self._loss_name + "=" + str(self._loss_stats)
                        )
                        log_data[self._loss_name] = self._loss_stats
                    
                    log_data["iter_per_sec"] = pbar.format_dict["rate"] if pbar.format_dict["rate"] is not None else -1

                    # Execute callbacks before loggers because they can append to log_data
                    for callback in callbacks:
                        if not callback(step, log_data, self):
                            callback_stop = True

                    with timing.timed_scope(name="loggers"):
                        for logger in loggers:
                            logger(self.step_count, log_data, self.state)

                    if len(callbacks) > 0:
                        if mpi.mpi_any(callback_stop):
                            break

                    # Reset the timing of tqdm after the first step, to ignore compilation time
                    if first_step:
                        first_step = False
                        pbar.unpause()

                    # Update the progress bar
                    pbar.update(self.step_count - old_step)
                    old_step = self.step_count

                # Final update so that it shows up filled.
                pbar.update(self.step_count - old_step)

        if timeit:
            total_time = {"total_time": timer.total}
            for logger in loggers:
                # We use the final step count for this last log entry
                logger(self.step_count, total_time, self.state)

        # flush at the end of the evolution so that final values are saved to
        # file
        for logger in loggers:
            logger.flush(self.state)

        if timeit:
            self._timer = timer
            if self._is_root:
                print(timer)
        
        return loggers[0]