パラメータ推定
=================

0. カルマンフィルタの尤度関数
-------------------------------

完全データX,Zが

:math:`l(\theta) = \ln p(X,Z|\theta) = \ln p(z_{0}|\mu_{0},P_{0})+ \sum_{n=2}^{N} \ln p(z_{n}|z_{n-1},F,Q)+ \sum_{n=1}^{N} \ln p(x_{n}|z_{n},H,R)`

:math:`\ln \mathcal{N} (\mathbf{x}|\mathbf{\mu},\mathbf{\Sigma}) = - \frac{D\ln(2 \pi)}{2}  -\frac{1}{2} \ln(|\Sigma|) -\frac{1}{2}(\mathbf{x}- \mathbf{\mu}) \Sigma^{-1}(\mathbf{x}- \mathbf{\mu})`


1. EMアルゴリズムを用いて求める
--------------------------------

パラメータ :math:`\theta = \lbrace A , \Gamma , C , \Sigma , \mu_{0} , P_{0} \rbrace` をEMアルゴリズムによって求める


* 完全データの対数尤度関数

  :math:`\ln p(\mathbf{X},\mathbf{Z}|\theta) = \ln(z_{1}|\mu_{0},P_{0}) + \sum_{n=2}^N \ln p(z_{n}|z_{n-1},A,\Gamma) + \sum_{n=1}^{N} \ln p(x_{n}|z_{n},C,\Sigma)`

* 完全データの尤度関数の期待値

  :math:`Q(\theta,\theta^{old})=E_{\mathbf{Z}|\theta^{old}}[\ln p(\mathbf{X},\mathbf{Z}|\theta)] = \sum_{\mathbf{Z}} \ln p(\mathbf{X},\mathbf{Z}|\theta) p(\mathbf{Z}|\mathbf{X},\theta^{old})`

E-step
-------
E-stepにおける隠れ変数の分布はカルマンスムーザーの方法を用いる

* :math:`p(\mathbf{Z}|\mathbf{X},\theta^{old})`

* :math:`J_{n} = V_{n} A^{T} (P_{n})^{-1}`

* :math:`\hat{\mu}_{n} = \mu_{n} + J_{n}(\hat{\mu}_{n+1} - A \mu_{n})`

* :math:`\hat{V}_{n} = V_{n} + J_{n}(\hat{V}_{n+1} -P_{n})J_{n}^{T}`

* :math:`E[z_{n}] = \hat{\mu_{n}}`

* :math:`E[z_{n} z_{n-1}^{T}] = \hat{V}_{n} J_{n-1}^{T} + \hat{\mu}_{n} \hat{\mu}_{n-1}^{T}`

* :math:`E[z_{n} z_{n}^{T}] = \hat{V}_{n} + \hat{\mu}_{n} \hat{\mu}_{n}^{T}`


M-step
----------

* :math:`\mu_{0}^{new} = E[z_{1}]`

* :math:`P_{0}^{new} = E[z_{1} z_{1}^{T}] - E[z_{1}]E[z_{1}^{T}]`

* :math:`A^{new} =(\sum_{n=2}^{N} E[z_{n} z_{n-1}^{T}])(\sum_{n=2}^{N} E[z_{n-1} z_{n-1}^{T}])^{-1}`

* :math:`\Gamma^{new} = \frac{1}{N-1} \sum_{n=2}^{N}(E[z_{n}z_{n}^{T}] - A ^{new} E[z_{n-1}z_{n}^{T}]-E[z_{n} z_{n-1}^{T}](A^{new})^{T} + A^{new} E[z_{n-1} z_{n-1}^{T}](A^{new})^{T})`

* :math:`C^{new} = (\sum_{n=1}^{N}x_{n}E[z_{n}^{T}])(\sum_{n=1}^{N}E[z_{n}z_{n}^{T}])^{-1}`

* :math:`\Sigma^{new} = \frac{1}{N} \sum_{n=1}^{N} (x_{n}x_{n}^{T} -C^{new} E[z_{n}]x_{n}^{T} -x_{n}E[z_{n}^{T}](C^{new})^{T}+C^{new}E[z_{n}z_{n}^{T}](C^{new})^{T})`


2. MCMC法により求める
------------------------
