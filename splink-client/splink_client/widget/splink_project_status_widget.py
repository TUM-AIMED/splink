"""
    A widget to show the progress and status of the sPLINK project

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

from hyfed_client.util.gui import add_labels
from hyfed_client.widget.hyfed_project_status_widget import HyFedProjectStatusWidget


class SplinkProjectStatusWidget(HyFedProjectStatusWidget):
    """ This widget implements sPLINK specific project status/progress labels such as chunk number """

    def __init__(self, title, project):

        super().__init__(title, project)

        # label to show the current chunk number; re-initialized in the add_splink_labels function
        self.chunk_label = None

    def add_splink_labels(self):
        """ Add chunk label """

        self.chunk_label = add_labels(widget=self, left_label_text="Chunk:",
                                      right_label_text=self.project.get_chunk_text())

    def update_splink_labels(self):
        """ Update the value of the chunk label """

        self.chunk_label.configure(text=self.project.get_chunk_text())

    def update_status_widget(self):
        """ OVERRIDDEN: update sPLINK project status widget (e.g. progress, splink, and status labels) """

        try:
            self.update_progress_labels()
            self.update_status_labels()
            self.update_splink_labels()

            self.after(500, self.update_status_widget)  # update labels every 500 ms
        except Exception as exp:
            pass

