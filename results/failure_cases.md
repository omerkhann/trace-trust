Representative cases from the SWE-Gym test run. Each is the run nearest its group's median final score, among runs of at most 25 actions (fixed rule, not hand-picked). Columns: actions so far, phase, p(resolved), decision, action, target. Full logs: logs/sample/.

### Success 1: failing run with a real patch, correctly rejected

Run `s1r1300`, task `modin-project__modin-7089`: REJECT after 20/20 actions; actually failed. 26 patch lines.

```
  1 locate         - CONTINUE        view    /workspace/modin-project__modin__0.28
  2 locate         - CONTINUE        view    /workspace/modin-project__modin__0.28
  3 locate         - CONTINUE        view    /workspace/modin-project__modin__0.28
  4 locate         - INJECT_HINT     view    /workspace/modin-project__modin__0.28
    -> pattern repeat_loop
    hint: You have run the same action several times without new results. Stop repeating it, state what you learned, and try a different approach.
  5 locate         - CONTINUE        bash    ls -R /workspace/modin-project__modin__0.28
  6 locate     0.079 CONTINUE        bash    grep -r 'def rank' /workspace/modin-project__modin__0.28/mod
  7 locate     0.079 CONTINUE        view    /workspace/modin-project__modin__0.28/modin/pandas/base.py
  8 locate     0.079 CONTINUE        bash    grep -n 'def rank' /workspace/modin-project__modin__0.28/mod
  9 locate     0.079 CONTINUE        view    /workspace/modin-project__modin__0.28/modin/pandas/base.py
 10 fix        0.079 CONTINUE        edit    /workspace/modin-project__modin__0.28/modin/pandas/base.py error
 11 fix        0.082 CONTINUE        edit    /workspace/modin-project__modin__0.28/modin/pandas/base.py
 12 fix        0.082 CONTINUE        create  /workspace/modin-project__modin__0.28/test_rank_error.py
 13 verify     0.082 CONTINUE        bash    python /workspace/modin-project__modin__0.28/test_rank_error error
 14 fix        0.082 CONTINUE        create  /workspace/modin-project__modin__0.28/test_rank_error_dask.p
 15 verify     0.082 CONTINUE        bash    python /workspace/modin-project__modin__0.28/test_rank_error error
 16 verify     0.090 CONTINUE        message 
 17 verify     0.090 CONTINUE        message 
 18 verify     0.090 CONTINUE        message 
 19 verify     0.090 CONTINUE        message 
 20 verify     0.090 CONTINUE        message 
 20 verify     0.045 REJECT                  (end of run)
    top factors: patch_empty +1.80, patch_lines -0.51, phase_entropy +0.35
    -> final p(resolved) 0.045 below trust threshold 0.080
```

### Success 2: passing run, correctly sent to a reviewer

Run `s2r1410`, task `Project-MONAI__MONAI-1884`: FLAG_FOR_REVIEW after 13/13 actions; actually resolved. 31 patch lines.

```
  1 locate         - CONTINUE        view    /workspace/Project-MONAI__MONAI__0.4
  2 locate         - CONTINUE        view    /workspace/Project-MONAI__MONAI__0.4/monai/transforms/utilit
  3 locate         - CONTINUE        bash    grep -n 'class ConcatItemsd' /workspace/Project-MONAI__MONAI
  4 locate         - CONTINUE        view    /workspace/Project-MONAI__MONAI__0.4/monai/transforms/utilit
  5 locate         - CONTINUE        view    /workspace/Project-MONAI__MONAI__0.4/monai/transforms/utilit
  6 fix        0.162 CONTINUE        edit    /workspace/Project-MONAI__MONAI__0.4/monai/transforms/utilit
  7 fix        0.162 CONTINUE        create  /workspace/Project-MONAI__MONAI__0.4/test_concat_itemsd.py
  8 verify     0.162 CONTINUE        bash    python /workspace/Project-MONAI__MONAI__0.4/test_concat_item error
  9 fix        0.162 CONTINUE        edit    /workspace/Project-MONAI__MONAI__0.4/test_concat_itemsd.py
 10 verify     0.162 CONTINUE        bash    python /workspace/Project-MONAI__MONAI__0.4/test_concat_item
 11 verify     0.137 CONTINUE        message 
 12 verify     0.137 CONTINUE        message 
 13 submit     0.137 CONTINUE        finish  finish
 13 submit     0.245 FLAG_FOR_REVIEW         (end of run)
    top factors: patch_empty +1.80, patch_lines -0.58, n_distinct_tools +0.57
    -> final p(resolved) 0.245 at or above trust threshold 0.080
```

### Success 3: failing run, correctly aborted early

Run `s0r1005`, task `iterative__dvc-5004`: ABORT after 5/22 actions; actually failed. 20 patch lines.

```
  1 locate         - CONTINUE        view    /workspace/iterative__dvc__1.10
  2 locate         - CONTINUE        view    /workspace/iterative__dvc__1.10 error
  3 locate         - CONTINUE        view    /workspace/iterative__dvc__1.10
  4 locate         - CONTINUE        view    /workspace/iterative__dvc__1.10 error
  5 locate         - CONTINUE        view    /workspace/iterative__dvc__1.10
  5 locate     0.037 ABORT           view    /workspace/iterative__dvc__1.10/dvc
    -> p(resolved) 0.037 below abort threshold 0.060 after 5 actions
```

### Failure 1: missed failure, largest category (M1 looks verified)

Run `s2r1800`, task `pandas-dev__pandas-53682`: FLAG_FOR_REVIEW after 19/19 actions; actually failed. 11 patch lines.

```
  1 locate         - CONTINUE        view    /workspace/pandas-dev__pandas__2.1
  2 locate         - CONTINUE        view    /workspace/pandas-dev__pandas__2.1 error
  3 locate         - CONTINUE        view    /workspace/pandas-dev__pandas__2.1
  4 locate         - CONTINUE        bash    ls -R /workspace/pandas-dev__pandas__2.1
  5 reproduce      - CONTINUE        create  /workspace/pandas-dev__pandas__2.1/reproduce_error.py
  6 reproduce  0.109 CONTINUE        bash    python /workspace/pandas-dev__pandas__2.1/reproduce_error.py error
  7 reproduce  0.109 CONTINUE        bash    python /workspace/pandas-dev__pandas__2.1/reproduce_error.py error
  8 reproduce  0.109 CONTINUE        bash    ls /workspace/pandas-dev__pandas__2.1/reproduce_error.py error
  9 fix        0.109 CONTINUE        create  /workspace/pandas-dev__pandas__2.1/reproduce_error.py
 10 verify     0.109 CONTINUE        bash    python /workspace/pandas-dev__pandas__2.1/reproduce_error.py
 11 locate     0.082 CONTINUE        view    /workspace/pandas-dev__pandas__2.1/pandas/_libs/lib.pyx
 12 locate     0.082 CONTINUE        bash    grep -n 'maybe_convert_objects' /workspace/pandas-dev__panda
 13 locate     0.082 CONTINUE        view    /workspace/pandas-dev__pandas__2.1/pandas/_libs/lib.pyx
 14 locate     0.082 CONTINUE        view    /workspace/pandas-dev__pandas__2.1/pandas/_libs/lib.pyx
 15 fix        0.082 CONTINUE        edit    /workspace/pandas-dev__pandas__2.1/pandas/_libs/lib.pyx
 16 verify     0.090 CONTINUE        bash    python /workspace/pandas-dev__pandas__2.1/reproduce_error.py
 17 verify     0.090 CONTINUE        message 
 18 verify     0.090 CONTINUE        message 
 19 submit     0.090 CONTINUE        finish  finish
 19 submit     0.212 FLAG_FOR_REVIEW         (end of run)
    top factors: patch_empty +1.80, phase_entropy +0.60, n_distinct_tools +0.57
    -> final p(resolved) 0.212 at or above trust threshold 0.080
```

### Failure 2: passing run wrongly rejected

Run `s0r1337`, task `getmoto__moto-7111`: REJECT after 16/16 actions; actually resolved. 29 patch lines.

```
  1 locate         - CONTINUE        view    /workspace/getmoto__moto__4.2 error
  2 locate         - CONTINUE        view    /workspace/getmoto__moto__4.2
  3 explore        - CONTINUE        view    /workspace/getmoto__moto__4.2/moto
  4 locate         - CONTINUE        bash    grep -rn 'def create_function' /workspace/getmoto__moto__4.2
  5 locate         - CONTINUE        view    /workspace/getmoto__moto__4.2/moto/awslambda/models.py
  6 fix        0.176 CONTINUE        edit    /workspace/getmoto__moto__4.2/moto/awslambda/models.py
  7 fix        0.176 CONTINUE        create  /workspace/getmoto__moto__4.2/reproduce_error.py
  8 verify     0.176 CONTINUE        bash    python /workspace/getmoto__moto__4.2/reproduce_error.py error
  9 fix        0.176 CONTINUE        edit    /workspace/getmoto__moto__4.2/reproduce_error.py
 10 fix        0.176 CONTINUE        edit    /workspace/getmoto__moto__4.2/reproduce_error.py
 11 verify     0.152 CONTINUE        bash    python /workspace/getmoto__moto__4.2/reproduce_error.py error
 12 fix        0.152 CONTINUE        edit    /workspace/getmoto__moto__4.2/reproduce_error.py error
 13 fix        0.152 CONTINUE        edit    /workspace/getmoto__moto__4.2/reproduce_error.py
 14 fix        0.152 CONTINUE        edit    /workspace/getmoto__moto__4.2/reproduce_error.py
 15 fix        0.152 CONTINUE        edit    /workspace/getmoto__moto__4.2/reproduce_error.py
 16 fix        0.090 INJECT_HINT     edit    /workspace/getmoto__moto__4.2/reproduce_error.py
    -> pattern repeat_loop
    hint: You have run the same action several times without new results. Stop repeating it, state what you learned, and try a different approach.
 16 fix        0.034 REJECT                  (end of run)
    top factors: patch_empty +1.80, patch_lines -0.55, phase_entropy +0.39
    -> final p(resolved) 0.034 below trust threshold 0.080
```

### Failure 3: passing run wrongly aborted

Run `s0r990`, task `getmoto__moto-7111`: ABORT after 5/23 actions; actually resolved. 54 patch lines.

```
  1 locate         - CONTINUE        view    /workspace/getmoto__moto__4.2
  2 locate         - CONTINUE        view    /workspace/getmoto__moto__4.2
  3 explore        - CONTINUE        view    /workspace/getmoto__moto__4.2/
  4 explore        - CONTINUE        view    /workspace/getmoto__moto__4.2/ error
  5 explore        - CONTINUE        view    /workspace/getmoto__moto__4.2/
  5 explore    0.052 ABORT           bash    ls -R /workspace/getmoto__moto__4.2
    -> p(resolved) 0.052 below abort threshold 0.060 after 5 actions
```
