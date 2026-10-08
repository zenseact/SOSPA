import copy
import unittest
import numpy as np
import sys
import os

from sospa_eval.order_polylines import (
    to_polyline_type,
)
from sospa_eval.types import Polyline
     
class TestToPolylineType(unittest.TestCase):

    def test_open_polyline_returns_not_closed(self):
        polyline = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        result = to_polyline_type(polyline)
        self.assertIsInstance(result, Polyline)
        self.assertFalse(result.is_closed)
        
        np.testing.assert_array_equal(result.geometry, polyline)
        self.assertEqual(len(result.geometry), len(polyline))
        
    def test_closed_polyline_removes_duplicate_end_point(self):
        polyline = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
        result = to_polyline_type(polyline)
        self.assertIsInstance(result, Polyline)
        self.assertTrue(result.is_closed)
        
        np.testing.assert_array_equal(result.geometry, polyline[:-1])
        self.assertEqual(len(result.geometry), len(polyline) - 1)

    def test_two_point_polyline_always_open(self):
        # _is_shape_closed returns False for len <= 2
        polyline = np.array([[0.0, 0.0], [0.0, 0.0]])
        result = to_polyline_type(polyline)
        self.assertFalse(result.is_closed)
        np.testing.assert_array_equal(result.geometry, polyline)

    def test_single_point_polyline_is_open(self):
        polyline = np.array([[1.0, 2.0]])
        result = to_polyline_type(polyline)
        self.assertFalse(result.is_closed)
        np.testing.assert_array_equal(result.geometry, polyline)


if __name__ == '__main__':
    unittest.main()