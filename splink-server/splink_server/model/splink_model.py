"""
    Model class for sPLINK project specific fields

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

from django.db import models
from hyfed_server.model.hyfed_models import HyFedProjectModel


class SplinkProjectModel(HyFedProjectModel):
    """
        The model inherits from HyFedProjectModel
        so it implicitly has id, name, status, etc, fields defined in the parent model
    """

    covariates = models.CharField(max_length=512, default="")
    chunk_size = models.PositiveIntegerField(default=10)
    max_iterations = models.PositiveIntegerField(default=20)
