import pytest
from py_checker import initialize, check

@pytest.fixture(scope="session") #전체 테스트에서 1번만 초기화
def setup():
  initialize()


@pytest.mark.parametrize("route", [
  [4, 13, 14],
  [14, 13, 4]
])

def test_success_cases(setup, route):
  result = check(route)
  assert result == True

# def test_success_case(setup):
#   result = check([4, 13, 14])
#   assert result == True

def test_fail_case(setup):
  result = check([1, 2, 3, 4, 5])
  assert result == False

# pytest 사용방법
# cd gopt
# pytest test_py_checker.py -v