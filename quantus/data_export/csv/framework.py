import pandas as pd
from typing import List

from ...data_objs.visualizations import ParamapDrawingBase
from ...data_objs.data_export import BaseDataExport
from ..options import get_data_export_types

class CSVExport(BaseDataExport):
    """Export parametric map data to CSV format.
    """
    def __init__(self, visualizations_obj: ParamapDrawingBase, output_path: str, function_names: List[str],
                 **kwargs):
        super().__init__(visualizations_obj, output_path)
        self.function_names = function_names
        _, self.functions = get_data_export_types() 
        self.kwargs = kwargs
        assert output_path.endswith(".csv"), "Output path must end with .csv to export to CSV format."
        
    def save_data(self):
        """Data saved dynamically to a dict object and eventually converted to a pandas dataframe and saved to a CSV file.
        It is recommended to only save numerical or string data types to avoid issues with CSV format. For more 
        complex data types (e.g. lists, dicts), consider using the PKLExport option instead.
        """
        data_dict = super().save_data()
        
        for function_name in self.function_names:
            function = self.functions["csv"][function_name]
            function(self.visualizations_obj, data_dict, **self.kwargs)
            
        if len(self.function_names):
            # Check if data_dict contains only scalar values (lists with single elements)
            all_scalar = True
            for key, value in data_dict.items():
                if not isinstance(value, list) or len(value) != 1:
                    all_scalar = False
                    break
            
            if all_scalar and data_dict:
                # If all values are scalar, create DataFrame with index [0]
                self.exported_df = pd.DataFrame(data_dict, index=[0])
            else:
                # Otherwise, create DataFrame normally
                self.exported_df = pd.DataFrame(data_dict)
            
            self.exported_df.to_csv(self.export_path, index=False)
        else:
            print("No CSV data exported. No export functions provided.")
        