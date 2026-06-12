"""Minimal tests for RoboTwin task pose adapters.

Run with:
    python -m robotwin_birelpose.tests.test_task_adapters
"""

import numpy as np

from robotwin_birelpose.task_adapters import (
    extract_object_poses_from_task,
    extract_target_pose_from_task,
    get_attr_pose,
    pose_from_sapien_pose,
)


class FakePose:
    def __init__(self, p, q):
        self.p = np.asarray(p, dtype=np.float64)
        self.q = np.asarray(q, dtype=np.float64)


class FakeActor:
    def __init__(self, pose):
        self._pose = pose

    def get_pose(self):
        return self._pose


class FakeTask:
    def __init__(self):
        self.object = FakeActor(FakePose([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0]))
        self.target_object = FakeActor(FakePose([4.0, 5.0, 6.0], [0.0, 1.0, 0.0, 0.0]))
        self.bottle1 = FakeActor(FakePose([0.1, 0.2, 0.3], [1.0, 0.0, 0.0, 0.0]))
        self.bottle2 = FakeActor(FakePose([0.4, 0.5, 0.6], [0.0, 1.0, 0.0, 0.0]))
        self.left_target_pose = [-0.06, -0.105, 1.0, 0.0, 1.0, 0.0, 0.0]
        self.right_target_pose = [0.06, -0.105, 1.0, 0.0, 1.0, 0.0, 0.0]
        self.box = FakeActor(FakePose([0.7, 0.8, 0.9], [1.0, 0.0, 0.0, 0.0]))
        self.target_box = FakeActor(FakePose([0.2, 0.3, 0.4], [1.0, 0.0, 0.0, 0.0]))


def test_pose_from_sapien_pose():
    pose = FakePose([1.0, 2.0, 3.0], [0.0, 1.0, 0.0, 0.0])
    pose7 = pose_from_sapien_pose(pose)

    np.testing.assert_allclose(pose7, [1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 0.0])


def test_get_attr_pose():
    task = FakeTask()
    pose7 = get_attr_pose(task, "object")

    np.testing.assert_allclose(pose7, [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])


def test_extract_place_a2b_right():
    task = FakeTask()
    object_result = extract_object_poses_from_task(task, "place_a2b_right")
    target_result = extract_target_pose_from_task(task, "place_a2b_right")

    assert object_result["object_name"] == "object"
    assert target_result["target_name"] == "target_object"
    np.testing.assert_allclose(object_result["object_pose"][:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(target_result["target_pose"][:3], [4.0, 5.0, 6.0])


def test_extract_pick_dual_bottles():
    task = FakeTask()
    object_result = extract_object_poses_from_task(task, "pick_dual_bottles", object_index=1)
    target_result = extract_target_pose_from_task(task, "pick_dual_bottles", target_index=1)

    assert object_result["object_name"] == "bottle2"
    assert set(object_result["all_object_poses"]) == {"bottle1", "bottle2"}
    assert target_result["target_name"] == "right_target_pose"
    np.testing.assert_allclose(object_result["object_pose"][:3], [0.4, 0.5, 0.6])
    np.testing.assert_allclose(target_result["target_pose"], task.right_target_pose)


def test_extract_handover_block():
    task = FakeTask()
    object_result = extract_object_poses_from_task(task, "handover_block")
    target_result = extract_target_pose_from_task(task, "handover_block")

    assert object_result["object_name"] == "box"
    assert target_result["target_name"] == "target_box"
    np.testing.assert_allclose(object_result["object_pose"][:3], [0.7, 0.8, 0.9])
    np.testing.assert_allclose(target_result["target_pose"][:3], [0.2, 0.3, 0.4])


def main():
    test_pose_from_sapien_pose()
    test_get_attr_pose()
    test_extract_place_a2b_right()
    test_extract_pick_dual_bottles()
    test_extract_handover_block()
    print("test_task_adapters: all tests passed")


if __name__ == "__main__":
    main()
