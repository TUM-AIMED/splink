"""
    Utility functions

    Copyright 2021 Reza NasiriGerdeh and Reihaneh TorkzadehMahani. All Rights Reserved.

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

import numpy as np

DIGITS_OF_PRECISION = 4


def round_result(result):
    """ Round result by customized number of digits of precision according to the value of the result """

    try:
        if (type(result) == np.float32 or type(result) == np.float64) and result != 0.0:
            if result >= 1.0:
                digits_of_precision = DIGITS_OF_PRECISION
            else:
                digits_of_precision = int(-np.log10(np.abs(result))) + DIGITS_OF_PRECISION

            return np.round(result, digits_of_precision)
        else:
            return result
    except:
        return result
