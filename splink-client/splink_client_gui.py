"""
    sPLINK client GUI

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

# to prevent numpy from using all cores
import os
os.environ['OPENBLAS_MAIN_FREE'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from hyfed_client.widget.join_widget import JoinWidget
from hyfed_client.util.hyfed_parameters import HyFedProjectParameter, ConnectionParameter, AuthenticationParameter

from splink_client.widget.splink_project_info_widget import SplinkProjectInfoWidget
from splink_client.widget.splink_dataset_widget import SplinkDatasetWidget
from splink_client.project.splink_client_project import SplinkClientProject
from splink_client.util.splink_parameters import SplinkProjectParameter
from splink_client.widget.splink_project_status_widget import SplinkProjectStatusWidget

import threading

import logging

logger = logging.getLogger(__name__)


class SplinkClientGUI:
    """ sPLINK Client GUI """

    def __init__(self):

        # create the join widget
        self.join_widget = JoinWidget(title="sPLINK Client",
                                      local_server_name="Localhost",
                                      local_server_url="http://localhost:8000",
                                      local_compensator_name="Localhost",
                                      local_compensator_url="http://localhost:8001",
                                      external_server_name="sPLINK-Server",
                                      external_server_url="https://exbio.wzw.tum.de/splink/api",
                                      external_compensator_name="HyFed-SDU",
                                      external_compensator_url="https://compensator.compbio.sdu.dk",
                                      )

        # show the join widget
        self.join_widget.show()

        # if join was NOT successful, terminate the client GUI
        if not self.join_widget.is_joined():
            return

        # if join was successful, get connection and authentication parameters from the join widget
        connection_parameters = self.join_widget.get_connection_parameters()
        authentication_parameters = self.join_widget.get_authentication_parameters()

        #  create sPLINK project info widget based on the authentication and connection parameters
        self.splink_project_info_widget = SplinkProjectInfoWidget(title="sPLINK Project Info",
                                                                  connection_parameters=connection_parameters,
                                                                  authentication_parameters=authentication_parameters)

        # Obtain sPLINK project info from the server
        # the project info will be available in project_parameters attribute of the info widget
        self.splink_project_info_widget.obtain_project_info()

        # if sPLINK project info cannot be obtained from the server, exit the GUI
        if not self.splink_project_info_widget.project_parameters:
            return

        # add basic info of the project such as project id, project name, description, and etc to the info widget
        self.splink_project_info_widget.add_project_basic_info()

        # add sPLINK specific project info to the widget
        self.splink_project_info_widget.add_splink_project_info()

        # add accept and decline buttons to the widget
        self.splink_project_info_widget.add_accept_decline_buttons()

        # show project info widget
        self.splink_project_info_widget.show()

        # if participant declined to proceed, exit the GUI
        if not self.splink_project_info_widget.is_accepted():
            return

        # if user agreed to proceed, create and show the sPLINK dataset selection widget
        project_parameters = self.splink_project_info_widget.get_project_parameters()
        covariates = project_parameters[SplinkProjectParameter.COVARIATES]
        self.splink_dataset_widget = SplinkDatasetWidget(title="sPLINK Dataset Selection",
                                                         covariates=covariates)
        self.splink_dataset_widget.add_quit_run_buttons()
        self.splink_dataset_widget.show()

        # if the participant didn't click on 'Run' button, terminate the client GUI
        if not self.splink_dataset_widget.is_run_clicked():
            return

        # if participant clicked on 'Run', get all the parameters needed
        # to create the client project from the widgets
        connection_parameters = self.join_widget.get_connection_parameters()
        authentication_parameters = self.join_widget.get_authentication_parameters()
        project_parameters = self.splink_project_info_widget.get_project_parameters()

        server_url = connection_parameters[ConnectionParameter.SERVER_URL]
        compensator_url = connection_parameters[ConnectionParameter.COMPENSATOR_URL]
        username = authentication_parameters[AuthenticationParameter.USERNAME]
        token = authentication_parameters[AuthenticationParameter.TOKEN]
        project_id = authentication_parameters[AuthenticationParameter.PROJECT_ID]

        algorithm = project_parameters[HyFedProjectParameter.ALGORITHM]
        project_name = project_parameters[HyFedProjectParameter.NAME]
        project_description = project_parameters[HyFedProjectParameter.DESCRIPTION]
        coordinator = project_parameters[HyFedProjectParameter.COORDINATOR]

        # sPLINK specific project info
        chunk_size = project_parameters[SplinkProjectParameter.CHUNK_SIZE]
        max_iterations = project_parameters[SplinkProjectParameter.MAX_ITERATIONS]
        covariates = project_parameters[SplinkProjectParameter.COVARIATES]
        dataset_file_path = self.splink_dataset_widget.get_dataset_file_path()
        phenotype_file_path = self.splink_dataset_widget.get_phenotype_file_path()
        covariate_file_path = self.splink_dataset_widget.get_covariate_file_path()
        phenotype_name = self.splink_dataset_widget.get_phenotype_name()
        cpu_cores = self.splink_dataset_widget.get_cpu_cores()

        # create sPLINK client project
        splink_client_project = SplinkClientProject(username=username,
                                                    token=token,
                                                    server_url=server_url,
                                                    compensator_url=compensator_url,
                                                    project_id=project_id,
                                                    algorithm=algorithm,
                                                    name=project_name,
                                                    description=project_description,
                                                    coordinator=coordinator,
                                                    result_dir='.',
                                                    log_dir='.',
                                                    dataset_file_path=dataset_file_path,
                                                    phenotype_file_path=phenotype_file_path,
                                                    covariate_file_path=covariate_file_path,
                                                    covariate_names=covariates,
                                                    phenotype_name=phenotype_name,
                                                    chunk_size=chunk_size,
                                                    max_iterations=max_iterations,
                                                    cpu_cores=cpu_cores)

        # run sPLINK client project as a thread
        splink_project_thread = threading.Thread(target=splink_client_project.run)
        splink_project_thread.setDaemon(True)
        splink_project_thread.start()

        # create and show sPLINK project status widget
        splink_project_status_widget = SplinkProjectStatusWidget(title="sPLINK Project Status",
                                                                 project=splink_client_project)
        splink_project_status_widget.add_static_labels()
        splink_project_status_widget.add_progress_labels()
        splink_project_status_widget.add_splink_labels()
        splink_project_status_widget.add_status_labels()
        splink_project_status_widget.add_log_and_quit_buttons()
        splink_project_status_widget.show()


if __name__ == "__main__":
    client_gui = SplinkClientGUI()
