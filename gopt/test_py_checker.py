import pytest
from py_checker import initialize, check, check_no_seq



# pytest 사용방법
# cd gopt
# pytest test_py_checker.py -v



@pytest.fixture(scope="session") #전체 테스트에서 1번만 초기화
def setup():
  initialize()


# 기존 check 테스트
@pytest.mark.parametrize("route", [
  [5, 9, 10, 15, 12],
  [1, 3, 8, 7, 14],
  [4, 13, 6]
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


# # check_no_seq 테스트
# @pytest.mark.parametrize("route", [
#   [5, 9, 10, 15, 12],
#   [1, 3, 8, 7, 14]
# ])

# def test_no_seq_success_cases(setup, route):
#   result = check_no_seq(route)
#   assert result == True

# def test_no_seq_fail_case(setup):
#   result = check_no_seq([1, 2, 3, 4, 5])
#   assert result == False