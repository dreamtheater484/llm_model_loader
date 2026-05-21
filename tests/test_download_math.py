import unittest


def percent(done, total):
    if not total:
        return 0
    return max(0, min(100, (done / total) * 100))


class DownloadMathTests(unittest.TestCase):
    def test_progress_math(self):
        self.assertEqual(percent(0, 100), 0)
        self.assertEqual(percent(50, 100), 50)
        self.assertEqual(percent(150, 100), 100)
        self.assertEqual(percent(10, 0), 0)


if __name__ == "__main__":
    unittest.main()

