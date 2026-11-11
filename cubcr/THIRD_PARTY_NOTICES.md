# Third-party notices
Files of this release that are derived from third-party code, with the upstream licence.  Third-party
code that is only downloaded at run time (for example by `rebuttal/scripts/e4_fetch_external.sh`) is not
redistributed here and keeps its own licence.

## `rebuttal/lib/e4_stratified_svarm.py` and `rebuttal/lib/e9_gpu_estimators.py`

Port of the reference implementation of Stratified SVARM (ApproxMethods/SVARM/StratifiedSVARM.py), used as a comparison estimator.  Upstream: `github.com/kolpaczki/Approximating-the-Shapley-Value-without-Marginal-Contributions`, commit `c4e30395b14b8bf9a8687d839d5b406f825bb23d`.  Upstream licence:

```
MIT License

Copyright (c) 2023 kolpaczki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
