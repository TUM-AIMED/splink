"""
    Mapper to map an algorithm name to the corresponding server project, model, and serializer

    Copyright 2021 Reza NasiriGerdeh. All Rights Reserved.

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

# HyFed project
from hyfed_server.project.hyfed_server_project import HyFedServerProject
from hyfed_server.model.hyfed_models import HyFedProjectModel
from hyfed_server.serializer.hyfed_serializers import HyFedProjectSerializer


# sPLINK project
from splink_server.project.splink_server_project import SplinkServerProject
from splink_server.model.splink_model import SplinkProjectModel
from splink_server.serializer.splink_serializers import SplinkProjectSerializer

# server_project, project_model, and project_serializer are mappers used in webapp_view
server_project = dict()
project_model = dict()
project_serializer = dict()


# HyFed project mapper values
hyfed_tool = 'HyFed'
server_project[hyfed_tool] = HyFedServerProject
project_model[hyfed_tool] = HyFedProjectModel
project_serializer[hyfed_tool] = HyFedProjectSerializer

# sPLINK tool mapper classes
stats_tool_name = 'sPLINK'
server_project[stats_tool_name] = SplinkServerProject
project_model[stats_tool_name] = SplinkProjectModel
project_serializer[stats_tool_name] = SplinkProjectSerializer


