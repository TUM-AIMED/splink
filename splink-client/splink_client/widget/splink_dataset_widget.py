"""
    sPLINK dataset widget to select the dataset file(s)

    Copyright 2021 Reihaneh TorkzadehMahani and Reza NasiriGerdeh. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

from hyfed_client.widget.hyfed_dataset_widget import HyFedDatasetWidget
from hyfed_client.util.gui import add_label_and_textbox, add_button, select_file_path
from tkinter import messagebox
import multiprocessing
import numpy as np


class SplinkDatasetWidget(HyFedDatasetWidget):
    """ This widget enables participants to select GWAS dataset files (i.e. bed file), phenotype, and covariate file """

    def __init__(self, title, covariates):

        super().__init__(title=title)

        self.covariates = covariates

        # dataset (.bed) file
        self.dataset_file_path = ''  # re-initialized in set_dataset_file_path function
        self.dataset_file_path_entry = add_label_and_textbox(widget=self, label_text='Dataset File (required)',
                                                             increment_row_number=False)
        add_button(widget=self, button_label="Browse", column_number=2, increment_row_number=True,
                   on_click_function=self.set_dataset_file_path)

        # phenotype file
        self.phenotype_file_path = ''  # re-initialized in set_phenotype_file_path function
        self.phenotype_file_path_entry = add_label_and_textbox(widget=self, label_text='Phenotype File (optional)',
                                                               increment_row_number=False)
        add_button(widget=self, button_label="Browse", column_number=2, increment_row_number=True,
                   on_click_function=self.set_phenotype_file_path)

        # phenotype column name
        self.phenotype_name = ''  # re-initialized in click_on_run function
        self.phenotype_name_entry = add_label_and_textbox(widget=self,
                                                          label_text='Phenotype Column Name \n'
                                                          '(required if phenotype file specified)',
                                                          increment_row_number=True)

        # covariate file
        self.covariate_file_path = ''  # re-initialized in set_covariate_file_path function
        if self.covariates:
            self.covariate_file_path_entry = add_label_and_textbox(widget=self,
                                                                   label_text='Covariate File',
                                                                   increment_row_number=False)
            add_button(widget=self, button_label="Browse", column_number=2, increment_row_number=True,
                       on_click_function=self.set_covariate_file_path)

        # textbox to specify the number of concurrent processes
        initial_cpu_cores = int(np.ceil(multiprocessing.cpu_count() / 2))
        self.cpu_cores_entry = add_label_and_textbox(widget=self, label_text="CPU Cores", increment_row_number=True,
                                                     value=initial_cpu_cores)
        self.cpu_cores = initial_cpu_cores  # re-initialized in click_on_run function

    def click_on_run(self):
        """ OVERRIDDEN: to ensure that participant selected required files or entered proper values  """

        if not self.dataset_file_path:
            messagebox.showerror('Error', 'Dataset file path cannot be empty!')
            return

        if self.phenotype_file_path and not self.phenotype_name_entry.get():
            messagebox.showerror('Error', 'Phenotype name cannot be empty!')
            return

        if self.covariates and not self.covariate_file_path:
            messagebox.showerror('Error', 'Covariate file path cannot be empty!')
            return

        if not self.cpu_cores_entry.get().isnumeric():
            messagebox.showerror('Error', 'CPU cores must be a positive integer!')
            return

        if int(self.cpu_cores_entry.get()) <= 0:
            messagebox.showerror('Error', 'CPU cores must be a positive integer!')
            return

        if int(self.cpu_cores_entry.get()) > multiprocessing.cpu_count():
            messagebox.showerror('Error', 'CPU cores must be less than the total CPU cores in the system!')
            return

        self.phenotype_name = self.phenotype_name_entry.get()
        self.cpu_cores = int(self.cpu_cores_entry.get())

        super().click_on_run()

    def set_dataset_file_path(self):
        self.dataset_file_path = select_file_path(self.dataset_file_path_entry, file_types=[('BED files', '*.bed')])

    def get_dataset_file_path(self):
        return self.dataset_file_path

    def set_phenotype_file_path(self):
        self.phenotype_file_path = select_file_path(self.phenotype_file_path_entry)

    def get_phenotype_file_path(self):
        return self.phenotype_file_path

    def set_covariate_file_path(self):
        self.covariate_file_path = select_file_path(self.covariate_file_path_entry)

    def get_covariate_file_path(self):
        return self.covariate_file_path

    def get_phenotype_name(self):
        return self.phenotype_name

    def get_cpu_cores(self):
        return self.cpu_cores
