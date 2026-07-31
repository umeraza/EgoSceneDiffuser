from egoscenediffuser.data.splits import subject_disjoint_split


def test_subject_split_is_deterministic_and_disjoint():
    subjects = [f"s{i}" for i in range(10)]
    first = subject_disjoint_split(subjects, (0.7, 0.15, 0.15), 42)
    second = subject_disjoint_split(subjects, (0.7, 0.15, 0.15), 42)
    assert first == second
    assert set(first) == set(subjects)
    assert set(first.values()) <= {"train", "val", "test"}
