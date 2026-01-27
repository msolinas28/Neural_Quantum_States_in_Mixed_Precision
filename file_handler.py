import pandas as pd
import os
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import json
import re
from typing import Dict

class Handler:
    def __init__(self, folder: str, model: str, arch: str, L: int, PBC: bool, n_dim: int, **kwargs):
        self.folder = folder
        self.model = model
        self.arch = arch
        self.L = L 
        self.PBC = PBC
        self.n_dim = n_dim
        self.kwargs = dict(sorted(kwargs.items()))

        if PBC: 
            self.BC = "PBC"
        else: 
            self.BC = "OBC"

    @property
    def list_name(self):
        return f"{self.model}_{self.arch}.csv"
    
    @property
    def _f_list_name(self):
        return self.folder + self.list_name
    
    @property
    def _exist_list(self):
        return os.path.exists(self._f_list_name)
    
    @property 
    def _keys(self):
        ks = ["BC", "n_dim", "L"]
        if self.kwargs is not None:
            for k in list(self.kwargs.keys()):
                ks.append(k)
        return ks
    
    @property
    def _values(self):
        vs = [self.BC, self.n_dim, self.L]
        if self.kwargs is not None:
            for v in list(self.kwargs.values()):
                vs.append(v)
        return vs
    
    @property
    def _dict_to_add(self):
        _dict_to_add = dict(zip(self._keys, self._values))
        return dict(sorted(_dict_to_add.items()))
    
    def exist_in_list(self):
        if self._exist_list:
            mask = self.df_list["tags"].apply(lambda x: x == self._dict_to_add)
            if sum(list(mask)):
                return list(self.df_list[mask].index)[0] + 1
            return 0
        else:
            raise ValueError(f"The following file does not exist: {self._f_list_name}")
        
    def _append_in_list(self):
        df = pd.concat([self.df_list, self._df_to_add], ignore_index=True)
        df['tags'] = df['tags'].apply(lambda x: json.dumps(x, sort_keys=True) if isinstance(x, dict) else x)
        df.to_csv(self._f_list_name, index=False)
            
    def _save_index(self):
        if self._exist_list:
            index = self.exist_in_list()
            if not index: 
                self._append_in_list()
                index = len(self.df_list)
            print("List updated!")
        else:
            self._create_list()
            index = 1
        return index

    def _load_index(self):
        index = self.exist_in_list()
        if not index:
            msg = "Parameters:\n" + "\n".join(f"    {k}: {v}" for k, v in self.kwargs.items())
            raise ValueError(
                    f"The selected {self._to_save_label} do not exist!\n"
                    f"  Architecture:      {self.arch}\n"
                    f"  Model:      {self.model}\n"
                    f"  {msg}\n"
                    f"  L:          {self.L}\n"
                    f"  n_dim:      {self.n_dim}\n"
                    f"  BC:         {self.BC}\n"
                    f"  Folder: {self.folder}"
                )
        return index

    def _create_list(self):
        if self._exist_list:
            raise ValueError("The list already exists!")
        
        os.makedirs(self.folder, exist_ok=True)
        self.df_list.to_csv(self._f_list_name, index=False)
        print("List Created!")
    
    @property
    def _df_to_add(self):
        return pd.DataFrame([{"tags": json.dumps(self._dict_to_add, sort_keys=True)}])

    @property
    def df_list(self):
        if self._exist_list:
            df = pd.read_csv(self._f_list_name)
            df["tags"] = df["tags"].apply(json.loads)
            return df
        else:
            return self._df_to_add
    
    @property
    def _to_save_label(self):
        return NotImplementedError
    
class VarHandler(Handler):
    @property
    def _to_save_label(self):
        return "variables"

    @property
    def _variables_folder(self):
        return self.folder + f"par_{self.model}_{self.arch}/"

    def save_variables(self, variables):
        index = self._save_index()
        folder = self._variables_folder
        binary_data = serialization.to_bytes(variables)
        os.makedirs(folder, exist_ok=True)
        with open(f"{folder}{str(index)}.mpack", "wb") as outfile:
            outfile.write(binary_data)
        print("Variables saved!")
    
    def load_parameters(self, variables):
        return self.load_variables(variables)["params"] 
    
    def load_variables(self, variables):
        with open(self.variables_file, "rb") as infile:
            binary_data = infile.read()
            variables = serialization.from_bytes(variables, binary_data)
            variables = jax.tree.map(lambda x: jnp.array(x), variables)
        return variables
    
    @property
    def variables_file(self):
        index = self._load_index()
        str_index = str(index)
        folder = self._variables_folder
        return f"{folder}{str_index}.mpack"
    
class HistoryHandler(Handler):
    @property
    def _to_save_label(self):
        return "history"
    
    @property
    def _history_folder(self):
        return self.folder + f"history_{self.model}_{self.arch}/"
    
    def save_history(self, log, timeit: bool=False, t=None):
        index = self._save_index()
        if "iter_per_sec" not in log.data and timeit:
            print("The given log does not have 'iter_per_sec' stored. Forcing 'timeit' to False")
            timeit = False

        folder = self._history_folder
        os.makedirs(folder, exist_ok=True)
        filepath = f"{folder}{str(index)}.csv"
        with open(filepath, "w") as f:
            if t is not None:
                for line in str(t).strip().split("\n"):
                    f.write(f"# {line}\n")  

        self._hist_df(log=log, timeit=timeit).to_csv(filepath, index=False, mode="a", header=True)
        print("History saved.")

    def load_history(self):
        return pd.read_csv(self.history_file, comment="#")

    def _hist_df(self, log, timeit: bool = False):
        data = {}
        for key in log["Energy"].keys():
            data[key] = log["Energy"][key]
        if timeit:
            data["iter_per_sec"] = log["iter_per_sec"]["value"]
            data["total_time"] = np.repeat(log["total_time"]["value"].item(), len(data["iter_per_sec"]))
        if "Condition_number" in log.data.keys():
            data["Condition_number"] = log["Condition_number"]["value"]
        if "sigma" in log.data.keys():
            data["sigma"] = log["sigma"]["value"]
        if "gradient_noise" in log.data.keys():
            data["gradient_noise"] = log["gradient_noise"]["value"]
        if "gradient_max" in log.data.keys():
            data["gradient_max"] = log["gradient_max"]["value"]

        return pd.DataFrame(data)
    
    @property
    def history_file(self):
        index = self._load_index()
        str_index = str(index)
        folder = self._history_folder
        return f"{folder}{str_index}.csv"
    
    @property
    def timing_info(self):
        timing_data = {}
        total_pattern = re.compile(r"Total:\s*([0-9.]+)")

        with open(self.history_file, 'r') as f:
            for line in f:
                if not line.strip().startswith('#'):
                    break

                total_match = total_pattern.search(line)
                if total_match:
                    timing_data['Total'] = float(total_match.group(1))
                    continue

                if '|' in line and ':' in line:
                    try:
                        main_part = line.split('|')[-1] 
                        name_str, time_str_full = main_part.split(':')
                        name = name_str.strip()
                        time_val = float(time_str_full.strip().split(' ')[0])
                        
                        timing_data[name] = time_val
                    except (ValueError, IndexError):
                        continue
        return timing_data

class GradHandler(Handler):
    @property
    def _to_save_label(self):
        return "gradients"

    @property
    def _gradients_folder(self):
        return self.folder + f"grad_{self.model}_{self.arch}/"

    def save_gradients_flattened(self, grad_flat):
        index = self._save_index()
        folder = self._gradients_folder
        filepath = f"{folder}{str(index)}.csv"
        os.makedirs(folder, exist_ok=True)
        np.savetxt(filepath, grad_flat, delimiter=",")
        print("Gradients saved!")
    
    def load_gradients_flattened(self):
        return np.loadtxt(self.variables_file, delimiter=",")
    
    @property
    def variables_file(self):
        index = self._load_index()
        str_index = str(index)
        folder = self._gradients_folder
        return f"{folder}{str_index}.csv"