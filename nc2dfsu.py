import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import xarray as xr
import numpy as np
from datetime import datetime
import os
import cftime
import pandas as pd

from mikecore.DfsuBuilder import DfsuBuilder
from mikecore.DfsuFile import eumUnit, DfsFile, DfsuFileType
from mikecore.DfsFactory import DfsFactory
from mikecore.DfsuFile import eumQuantity, eumUnit, eumItem

class NetCDFConverterGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("NetCDF to DFSU Converter")
        self.master.minsize(width=400, height=200)

        self.dataset = None  
        self.selected_variable = tk.StringVar() 

        self.frame_load = tk.Frame(self.master)
        self.frame_load.pack(padx=10, pady=5)

        self.frame_dimensions = tk.Frame(self.master)
        self.frame_dimensions.pack(padx=10, pady=5)

        self.frame_variables = tk.Frame(self.master)
        self.frame_variables.pack(padx=10, pady=5)

        self.load_file_btn = tk.Button(self.frame_load, text="Load NetCDF File", command=self.load_file)
        self.load_file_btn.pack()

    def load_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("NetCDF files", "*.nc"), ("All files", "*.*")])
        if self.file_path:
            try:
                self.dataset = xr.open_dataset(self.file_path)
                self.update_dimensions_display()
                self.update_variables_display()
                
                if not hasattr(self, 'frame_selectors'):
                    self.create_dimension_selectors()
                else:
                    self.update_dimension_selectors()

                if not hasattr(self, 'frame_convert'):
                    self.create_convert_button()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open file: {e}")

    def update_dimension_selectors(self):
        dimension_names = list(self.dataset.dims)

        self.x_selector['values'] = dimension_names
        self.y_selector['values'] = dimension_names
        self.z_selector['values'] = dimension_names

        self.set_default_dimension(self.x_selector, ['lon', 'longitude', 'X', 'x'])
        self.set_default_dimension(self.y_selector, ['lat', 'latitude', 'Y', 'y'])
        self.set_default_dimension(self.z_selector, ['depth', 'Depth', 'Z', 'z'], exclude='time')

    def update_dimensions_display(self):
        self.min_entries = {}  
        self.max_entries = {} 
        self.minimums = {}
        self.maximums = {}

        for widget in self.frame_dimensions.winfo_children():
            widget.destroy()

        tk.Label(self.frame_dimensions, text="Dimensions").grid(row=0, column=0, columnspan=5)
        row = 1

        for dim_name in self.dataset.dims:
            dimension = self.dataset[dim_name]
            min_val = self.format_value(dimension.min().values)
            max_val = self.format_value(dimension.max().values)

            tk.Label(self.frame_dimensions, text=f"{dim_name} range:").grid(row=row, column=0, sticky="ew")

            min_label = tk.Label(self.frame_dimensions, text="Minimum")
            min_label.grid(row=row, column=1)
            min_entry = tk.Entry(self.frame_dimensions)
            min_entry.grid(row=row, column=2)
            min_entry.insert(0, min_val)
            self.min_entries[dim_name] = min_entry
            if "time" in dim_name.lower():
                self.minimums[dim_name] = self.to_datetime(min_val)
            else:
                self.minimums[dim_name] = float(min_val)

            max_label = tk.Label(self.frame_dimensions, text="Maximum")
            max_label.grid(row=row, column=3)
            max_entry = tk.Entry(self.frame_dimensions)
            max_entry.grid(row=row, column=4)
            max_entry.insert(0, max_val)
            self.max_entries[dim_name] = max_entry
            if "time" in dim_name.lower():
                self.maximums[dim_name] = self.to_datetime(max_val)
            else:
                self.maximums[dim_name] = float(max_val)
            row = row + 1

    def get_dimension_ranges(self):
        dimension_ranges = {}
        for dim_name in self.min_entries.keys():
            min_val = self.min_entries[dim_name].get() 
            max_val = self.max_entries[dim_name].get() 
            if dim_name == "time":
                dimension_ranges[dim_name] = (self.to_datetime(min_val), self.to_datetime(max_val))
            else:
                min_val = float(min_val)
                max_val = float(max_val)
                dimension_ranges[dim_name] = (min_val, max_val)
            
        return dimension_ranges

    def on_combobox_keypress(self, event):
        if event.char.isalpha():
            current_combobox = event.widget
            typed_char = event.char.lower()

            for item in current_combobox['values']:
                if item.lower().startswith(typed_char):
                    current_combobox.set(item)
                    break

    def create_dimension_selectors(self):
        self.frame_selectors = tk.Frame(self.master)
        self.frame_selectors.pack(padx=10, pady=5)

        dimension_names = list(self.dataset.dims)

        tk.Label(self.frame_selectors, text="X Dimension:").grid(row=0, column=0, padx=5, pady=5)
        self.x_selector = ttk.Combobox(self.frame_selectors, values=dimension_names, state="readonly")
        self.x_selector.grid(row=0, column=1, padx=5, pady=5)
        self.set_default_dimension(self.x_selector, ['lon', 'longitude', 'X', 'x'])

        tk.Label(self.frame_selectors, text="Y Dimension:").grid(row=0, column=2, padx=5, pady=5)
        self.y_selector = ttk.Combobox(self.frame_selectors, values=dimension_names, state="readonly")
        self.y_selector.grid(row=0, column=3, padx=5, pady=5)
        self.set_default_dimension(self.y_selector, ['lat', 'latitude', 'Y', 'y'])

        tk.Label(self.frame_selectors, text="Z Dimension:").grid(row=0, column=4, padx=5, pady=5)
        self.z_selector = ttk.Combobox(self.frame_selectors, values=dimension_names, state="readonly")
        self.z_selector.grid(row=0, column=5, padx=5, pady=5)
        self.set_default_dimension(self.z_selector, ['depth', 'Depth', 'Z', 'z'], exclude='time')

    def create_convert_button(self):
        self.frame_convert = tk.Frame(self.master)
        self.frame_convert.pack(padx=10, pady=5)

        self.convert_btn = tk.Button(self.frame_convert, text="Convert", command=self.convert_file)
        self.convert_btn.pack()

    def set_default_dimension(self, selector, keywords, exclude=None):
        for dim in selector['values']:
            if any(keyword in dim for keyword in keywords) and (not exclude or exclude not in dim):
                selector.set(dim)
                break

    def update_variables_display(self):
        for widget in self.frame_variables.winfo_children():
            widget.destroy()

        tk.Label(self.frame_variables, text="Variables").grid(row=0, column=0, columnspan=5)

        variables = [var_name for var_name in self.dataset.variables if var_name not in self.dataset.dims]

        self.var_widgets = {}
        first_var = True

        items = [name for name in eumItem.__members__]
        itemsDisplay = [elem.replace("eumI", "") for elem in items]
        itemsDisplay.sort(key=str.lower)

        units = [name for name in eumUnit.__members__]
        unitsDisplay = [elem.replace("eumU", "") for elem in units]
        unitsDisplay.sort(key=str.lower)

        if variables:
            self.selected_variable.set(variables[0])

        row = 1
        for var_name in variables:
            rb = tk.Radiobutton(self.frame_variables, text=var_name, variable=self.selected_variable, value=var_name, command=self.update_variable_widgets)
            rb.grid(row=row, column=0)

            tk.Label(self.frame_variables, text="Item").grid(row=row, column=1)
            type_cb = ttk.Combobox(self.frame_variables, values=itemsDisplay, state="disabled")
            type_cb.bind("<KeyPress>", self.on_combobox_keypress)
            type_cb.grid(row=row, column=2)

            tk.Label(self.frame_variables, text="Unit").grid(row=row, column=3)
            unit_cb = ttk.Combobox(self.frame_variables, values=unitsDisplay, state="disabled")
            unit_cb.bind("<KeyPress>", self.on_combobox_keypress)
            unit_cb.grid(row=row, column=4)

            tk.Label(self.frame_variables, text="Name").grid(row=row, column=5)
            var_name_entry = tk.Entry(self.frame_variables, state="disabled")
            var_name_entry.grid(row=row, column=6)

            self.var_widgets[var_name] = (type_cb, unit_cb, var_name_entry)

            if first_var:
                self.selected_variable.set(var_name)
                type_cb.config(state="readonly")
                unit_cb.config(state="readonly")
                var_name_entry.config(state="normal")
                first_var = False
            row = row + 1

    def update_variable_widgets(self):
        selected_var = self.selected_variable.get()
        for var_name, widgets in self.var_widgets.items():
            type_cb, unit_cb, var_name_entry = widgets

            if var_name == selected_var:
                type_cb.config(state="readonly") 
                unit_cb.config(state="readonly") 
                var_name_entry.config(state="normal")
            else:
                type_cb.config(state="disabled")
                unit_cb.config(state="disabled")
                var_name_entry.config(state="disabled")




    def format_value(self, value):
        if isinstance(value, np.datetime64):
            return str(np.datetime_as_string(value, unit='s'))
        elif isinstance(value, float) or isinstance(value, np.ndarray):
            return f"{value:.2f}"
        else:
            return str(value)

    def _center2node(self, array):
        diff_array = np.diff(array)
        diff_array = np.append(diff_array[0], diff_array)
        diff_array = np.append(diff_array, diff_array[-1])
        arr = np.append(array, array[-1]+diff_array[-1])
        arr = arr - diff_array / 2
        return arr
    
    def _create_mesh_elements2d(self, x_range, y_range):
        x_range = np.asarray(x_range)
        y_range = np.asarray(y_range)
        nx = len(x_range)
        ny = len(y_range)
        points = np.array(np.meshgrid(x_range, y_range)).reshape(2, -1).T
        points = np.append(points, -np.ones(shape=(points.shape[0], 1)), axis=1)
        a = np.arange(nx * ny).reshape(ny, nx).T
        cells = np.array([a[:-1, :-1], a[1:, :-1], a[1:, 1:], a[:-1, 1:]]).reshape(4, -1).T
        return points, cells
    
    def _create_mesh_elements3d(self, x_range, y_range, z_range):
        x_range = np.asarray(x_range)
        y_range = np.asarray(y_range)
        z_range = np.asarray(z_range)
        nx1 = len(x_range)
        ny1 = len(y_range)
        nz1 = len(z_range)
        x, y, z = np.meshgrid(x_range, y_range, z_range)
        points = np.array([x.T, y.T, z.T]).T.reshape(-1, 3)
        a = np.arange(len(points)).reshape((ny1, nx1, nz1))
        a = np.transpose(a, [1, 0, 2])
        cells = (
            np.array(
                [
                    a[:-1, :-1, :-1],
                    a[1:, :-1, :-1],
                    a[1:, 1:, :-1],
                    a[:-1, 1:, :-1],
                    a[:-1, :-1, 1:],
                    a[1:, :-1, 1:],
                    a[1:, 1:, 1:],
                    a[:-1, 1:, 1:],
                ]
            ).reshape(8, -1).T
        )
        return points, cells

    def to_datetime(self, d):
        if isinstance(d, datetime):
            return d
        if isinstance(d, cftime.DatetimeNoLeap):
            return datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
        elif isinstance(d, cftime.DatetimeGregorian):
            return datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
        elif isinstance(d, str):
            errors = []
            for fmt in (
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%dT%H:%M:%SZ"):
                try:
                    return datetime.strptime(d, fmt)
                except ValueError as e:
                    errors.append(e)
                    continue
            raise Exception(errors)
        elif isinstance(d, np.datetime64):
            d = pd.to_datetime(d)
            return datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
        else:
            raise Exception("Unknown value: {} type: {}".format(d, type(d)))

    def _create_dfsu2D_file(self, fname, variable):
        x = self._center2node(self.lons)
        y = self._center2node(self.lats)
        points, cells = self._create_mesh_elements2d(x, y)
        cells = cells + 1
        codes = np.zeros(shape=points.shape[0], dtype=int)
        z = np.ones(shape=points.shape[0], dtype=float)
        
        time_original = self.dataset.time.values
        time = []
        for i, t in enumerate(time_original):
            time.append(self.to_datetime(t))

        points, cells = self._create_mesh_elements2d(x, y)
        cells = cells + 1

        if os.path.exists(fname):
            os.remove(fname)

        codes = np.zeros(shape=points.shape[0], dtype=int)
        z = np.ones(shape=points.shape[0], dtype=float)
        codes[(points[:, 0] == min(x)) | (points[:, 0] == max(x))] = 1
        codes[(points[:, 1] == min(y)) | (points[:, 1] == max(y))] = 1
        builder = DfsuBuilder.Create(DfsuFileType.Dfsu2D)
        builder.FileTitle = 'nc to Dfsu converter'
        factory = DfsFactory()
        builder.SetProjection(factory.CreateProjectionGeoOrigin('LONG/LAT', points[0][0], points[0][1], 0.0))
        builder.SetNodes(points[:, 0], points[:, 1], z, codes)
        builder.SetElements(cells)
        builder.DeleteValueByte = DfsFile.DefaultDeleteValueByte
        builder.DeleteValueDouble = DfsFile.DefaultDeleteValueDouble
        builder.DeleteValueFloat = DfsFile.DefaultDeleteValueFloat
        builder.DeleteValueInt = DfsFile.DefaultDeleteValueInt
        builder.DeleteValueUnsignedInt = DfsFile.DefaultDeleteValueUnsignedInt
        builder.SetZUnit(eumUnit.eumUmeter)
        builder.SetTemporalAxis(factory.CreateTemporalEqCalendarAxis(timeUnit=eumUnit.eumUsec, startDateTime=time[0], startTimeOffset=0, timeStep=24*60*60, numberOfTimeSteps=len(time), firstTimeStepIndex=0))
        
        builder.AddDynamicItem(self.item[0], self.item[1])
        
        dfsufile = builder.CreateFile(fname)
        for t in range(len(time)):
            d = self.dataset[variable].isel(time=t).load().values
            d[np.isnan(d)] = dfsufile.DeleteValueFloat
            d = d.astype(np.float32)
            dfsufile.WriteItemTimeStep(1, t, (time[t] - time[0]).total_seconds(), np.flip(d, axis=0).flatten('F'))
        dfsufile.Close()
                

    def _create_dfsu3D_file(self, fname, variable): 
        x = self._center2node(self.lons)
        y = self._center2node(self.lats)
        z = -np.flip(self._center2node(self.depths))
        
        time_original = self.dataset.time.values
        time = []
        for i, t in enumerate(time_original):
            time.append(self.to_datetime(t))

        points, cells = self._create_mesh_elements3d(x, y, z)
        cells = cells + 1

        if os.path.exists(fname):
            os.remove(fname)

        codes = np.zeros(shape=points.shape[0], dtype=int)
        codes[(points[:, 0] == min(x)) | (points[:, 0] == max(x))] = 1
        codes[(points[:, 1] == min(y)) | (points[:, 1] == max(y))] = 1
        builder = DfsuBuilder.Create(DfsuFileType.Dfsu3DSigma)
        builder.FileTitle = 'nc to Dfsu converter'
        factory = DfsFactory()
        builder.SetProjection(factory.CreateProjectionGeoOrigin('LONG/LAT', points[0][0], points[0][1], 0.0))
        builder.SetNodes(points[:, 0], points[:, 1], points[:, 2], codes)
        builder.SetElements(cells)
        builder.DeleteValueByte = DfsFile.DefaultDeleteValueByte
        builder.DeleteValueDouble = DfsFile.DefaultDeleteValueDouble
        builder.DeleteValueFloat = DfsFile.DefaultDeleteValueFloat
        builder.DeleteValueInt = DfsFile.DefaultDeleteValueInt
        builder.DeleteValueUnsignedInt = DfsFile.DefaultDeleteValueUnsignedInt
        builder.SetZUnit(eumUnit.eumUmeter)
        builder.SetNumberOfSigmaLayers(z.size - 1)
        builder.SetTemporalAxis(factory.CreateTemporalEqCalendarAxis(timeUnit=eumUnit.eumUsec, startDateTime=time[0], startTimeOffset=0, timeStep=24*60*60, numberOfTimeSteps=len(time), firstTimeStepIndex=0))
        builder.AddDynamicItem(self.item[0], self.item[1])
        dfsufile = builder.CreateFile(fname)
        zz = dfsufile.dfsFile.CreateEmptyItemData(1)
        zz.Data = points[:, 2].astype(np.float32)
        for t in range(len(time)):
            dfsufile.WriteItemTimeStep(1, t, (time[t] - time[0]).total_seconds(), zz.Data)
            d = self.dataset[variable].isel(time=t).load().values
            d[np.isnan(d)] = dfsufile.DeleteValueFloat
            d = d.astype(np.float32)
            dfsufile.WriteItemTimeStep(2, t, (time[t] - time[0]).total_seconds(), np.flip(d, axis=0).flatten('F'))
        dfsufile.Close()
        

    def convert_file(self):
        x_dim = self.x_selector.get()
        y_dim = self.y_selector.get()
        z_dim = self.z_selector.get()
        selected_var = self.selected_variable.get()

        type_cb, unit_cb, var_name_entry = self.var_widgets[selected_var]

        
        selected_type = getattr(eumItem, "eumI"+type_cb.get())
        selected_unit = getattr(eumUnit, "eumU"+unit_cb.get())
        selected_name = var_name_entry.get()

        self.item = [selected_name, eumQuantity.Create(selected_type, selected_unit)]

        dimension_ranges = self.get_dimension_ranges()
        
        ready = True
        for key in dimension_ranges.keys():
            if dimension_ranges[key][0] < self.minimums[key]:
                messagebox.showwarning("Warning", f"Minimum value of {key} is {self.minimums[key]}.")
                ready = False
            elif dimension_ranges[key][1] > self.maximums[key]:
                messagebox.showwarning("Warning", f"Maximum value of {key} is {self.maximums[key]}.")
                ready = False

        if ready:
            for key in dimension_ranges.keys():
                selection_dict = {key: slice(dimension_ranges[key][0], dimension_ranges[key][1])}
                self.dataset = self.dataset.sel(selection_dict)

            self.lons = self.dataset[x_dim].values
            self.lats = self.dataset[y_dim].values
            self.depths = self.dataset[z_dim].values
            self.dates = self.dataset.time.values.astype(datetime)

            fname = self.file_path.replace(".nc", " "+self.item[0]+".dfsu")
            if len(self.dataset[selected_var].dims) == 3:
                self._create_dfsu2D_file(fname, selected_var)
            else:
                self._create_dfsu3D_file(fname, selected_var)


if __name__ == "__main__":
    root = tk.Tk()
    app = NetCDFConverterGUI(root)
    root.mainloop()
