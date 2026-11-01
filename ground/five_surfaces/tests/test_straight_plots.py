"""Data gaps, material revisits and failure-safe telemetry exports."""
import asyncio
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'scripts'))
from plot_straight import surface_segments, values
from ground_export import plot_ground_run


class StraightPlotTests(unittest.TestCase):
    def test_material_revisit_is_not_concatenated_into_first_visit(self):
        rows = [dict(surface=s, time_s=i) for i,s in enumerate(
            ['Grass', 'Grass', 'DrySand', 'DrySand', 'Grass'])]
        self.assertEqual(surface_segments(rows), [('Grass',0,2),('DrySand',2,4),('Grass',4,5)])
        self.assertEqual(surface_segments([]), [])

    def test_invalid_slip_and_missing_stress_are_not_zero_filled(self):
        rows = [dict(left_slip_ratio=.4, left_slip_valid=0),
                dict(left_slip_ratio=-.3, left_slip_valid=1)]
        y = values(rows, 'left_slip_ratio', 100., 'left_slip_valid')
        self.assertTrue(math.isnan(y[0]))
        self.assertAlmostEqual(y[1], -30.)
        self.assertTrue(np.isnan(values(rows, 'soil_pressure_mean_pa')).all())

    def test_plot_failure_retains_csv_and_reports_error(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            original = 'time_s,left_slip_ratio\n0.1,nan\n'
            (directory/'physics.csv').write_text(original)
            (directory/'summary.json').write_text(json.dumps(dict(result='completed',plots_state='pending')))
            with patch('ground_export.asyncio.to_thread', new_callable=AsyncMock,
                       side_effect=RuntimeError('plot dependency unavailable')):
                summary = asyncio.run(plot_ground_run(directory))
            self.assertEqual(summary['plots_state'], 'failed')
            self.assertIn('plot dependency unavailable', summary['plot_error'])
            self.assertEqual((directory/'physics.csv').read_text(), original)
            self.assertEqual(json.loads((directory/'summary.json').read_text())['result'], 'completed')


if __name__ == '__main__':
    unittest.main()
