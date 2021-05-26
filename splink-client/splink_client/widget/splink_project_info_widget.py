"""
    A widget to add sPLINK-specific project info

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

from hyfed_client.widget.hyfed_project_info_widget import HyFedProjectInfoWidget
from hyfed_client.util.gui import add_label_and_textbox
from splink_client.util.splink_parameters import SplinkProjectParameter
from hyfed_client.util.hyfed_parameters import HyFedProjectParameter
from splink_client.util.splink_algorithms import SplinkAlgorithm


class SplinkProjectInfoWidget(HyFedProjectInfoWidget):
    def __init__(self, title, connection_parameters, authentication_parameters):

        super().__init__(title=title, connection_parameters=connection_parameters,
                         authentication_parameters=authentication_parameters)

    # sPLINK project specific info
    def add_splink_project_info(self):
        add_label_and_textbox(self, label_text="Chunk size",
                              value=str(self.project_parameters[SplinkProjectParameter.CHUNK_SIZE]) + 'K',
                              status='disabled')

        covariates = self.project_parameters[SplinkProjectParameter.COVARIATES]
        if covariates:
            add_label_and_textbox(self, label_text="Covariates",
                                  value=covariates,
                                  status='disabled')

        algorithm = self.project_parameters[HyFedProjectParameter.ALGORITHM]
        if algorithm == SplinkAlgorithm.LOGISTIC_REGRESSION:
            add_label_and_textbox(self, label_text="Max iterations",
                                  value=self.project_parameters[SplinkProjectParameter.MAX_ITERATIONS],
                                  status='disabled')
