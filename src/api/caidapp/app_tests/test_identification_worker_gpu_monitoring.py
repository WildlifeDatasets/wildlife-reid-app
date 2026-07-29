from django.test import TestCase

from caidapp import models, tasks


class IdentificationWorkerGpuMonitoringTest(TestCase):
    def test_records_unavailable_gpu_heartbeat(self):
        heartbeat_id = tasks.record_identification_worker_gpu_heartbeat.run(
            available=False,
            device="cuda:0",
            error_message="RuntimeError: CUDA-capable device is unavailable",
        )

        heartbeat = models.IdentificationWorkerGpuHeartbeat.objects.get(id=heartbeat_id)
        self.assertFalse(heartbeat.available)
        self.assertEqual(heartbeat.device, "cuda:0")
        self.assertIn("CUDA-capable", heartbeat.error_message)
        self.assertIsNotNone(heartbeat.recorded_at)

    def test_records_available_gpu_heartbeat(self):
        heartbeat_id = tasks.record_identification_worker_gpu_heartbeat.run(
            available=True,
            device="cuda:0",
            device_name="NVIDIA RTX",
            free_memory_gb=12.5,
            total_memory_gb=24.0,
        )

        heartbeat = models.IdentificationWorkerGpuHeartbeat.objects.get(id=heartbeat_id)
        self.assertTrue(heartbeat.available)
        self.assertEqual(heartbeat.device_name, "NVIDIA RTX")
        self.assertEqual(heartbeat.free_memory_gb, 12.5)
