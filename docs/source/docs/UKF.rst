Unscented Kalman Filter(UKF)
============================

Unscented Transform(UT)
-----------------------

UTはシグマポイントを用いて非線形変換後の確率変数を推定する手法です。

手法自体はモンテカルロ推定に似ていますが、アプローチの仕方は大きく異なります。


:math:`p(\mathbf{x}) = N(\mathbf{m},P)`

:math:`\mathbf{y} = \mathbf{g}(x)`

1. 2n+1 sigma points

  :math:`\chi ^ {0} = \mathbf{m}`

  :math:`\chi ^ {i} = \mathbf{m} + \sqrt{n+\lambda} \left[ \sqrt{P} \right]_{i}`

  :math:`\chi ^ {i + n} = \mathbf{m} - \sqrt{n+\lambda} \left[ \sqrt{P} \right]_{i}`

  i = 1,...,n

  :math:`\lambda` はスケールパラメータ,
  :math:`\lambda = \alpha^{2}(n+\kappa)-n`

2. propagate the sigma points

  :math:`y^{i}=\mathbf{g}(\chi ^ {i})`

  i = 0,...,2n

3. Estimate of the mean and covariance

  :math:`E[\mathbf{g}(x)] = \mu_{U} = \sum_{i=0}^{2n} {W_{i}^{(m)}y^{i}}`

  :math:`Cov[\mathbf{g}(x)] = S_{U} = \sum_{i=0}^{2n} {W_{i}^{(c)}(y^{i}-\mu_{U})(y^{i}-\mu_{U})^{T}}`

  :math:`W^{m}_{0}=\frac{\lambda}{n+\lambda}`,
  :math:`W^{c}_{0}=\frac{\lambda}{n+\lambda} +(1-\alpha^{2}+\beta)`,
  :math:`W^{m}_{i}=\frac{1}{2(n+\lambda)}`,
  :math:`W^{c}_{i}=\frac{1}{2(n+\lambda)}`

Unscented Kalman Filter(UKF)
--------------------------------

- Prediction

  1. calc sigma posints

    :math:`\chi ^ {(0))}_{k-1} = \mathbf{m}_{k-1}`

    :math:`\chi ^ {(i))}_{k-1} = \mathbf{m}_{k-1} + \sqrt{n+\lambda} \left[ \sqrt{P_{k-1}} \right]_{i}`

    :math:`\chi ^ {(i + n)}_{k-1} = \mathbf{m_{k-1}} - \sqrt{n+\lambda} \left[ \sqrt{P_{k-1}} \right]_{i}`

    i = 1,...,n

  2. propagete the sigma points through the dynamic model

    :math:`\chi^{i}_{k}=\mathbf{f}(\chi ^ {i}_{k-1})`

  3. Compute the predicted mean and the predicted covariance

    :math:`m_{k}^{-} = \sum_{i=0}^{2n} {W_{i}^{(m)} \chi_{k}^{i}}`

    :math:`P_{k}^{-} = \sum_{i=0}^{2n} {W_{i}^{(c)}(\chi_{k}^{i}-m_{k}^{-})(\chi_{k}^{i}-m_{k}^{-}}) + Q_{k-1}`

- Update

  1. calc sigma points

    :math:`\chi ^ {-(0))}_{k} = \mathbf{m}^{-}_{k}`

    :math:`\chi ^ {-(i))}_{k} = \mathbf{m}^{-}_{k} + \sqrt{n+\lambda} \left[ \sqrt{P_{k}} \right]_{i}`

    :math:`\chi ^ {-(i + n)}_{k} = \mathbf{m}^{-}_{k} - \sqrt{n+\lambda} \left[ \sqrt{P_{k}} \right]_{i}`

    i = 1,...,n

  2. propagete sigma points

    :math:`\hat{y}^{i}_{k}=\mathbf{h}(\chi ^ {-(i)}_{k})`

  3. compute the predicted mean,the predicted covariance of the meansurement and the cross covariance of the state and the measurement

    :math:`\mu_{k} = \sum_{i=0}^{2n} {W_{i}^{(m)} \hat{y}^{i}}`

    :math:`S_{k} = \sum_{i=0}^{2n} {W_{i}^{(c)}(\hat{y}_{k}^{i}-\mu_{k})(\hat{y}^{i}-\mu_{k})^{T}}+R_{k}`

    :math:`C_{k} = \sum_{i=0}^{2n} {W_{i}^{(c)}(\chi_{k}^{-(i)}-m^{-}_{k})(\hat{y}^{i}-\mu_{k})^{T}}`

  4. compute the filter gain,the filtered state mean and covariance

    :math:`K_{k}=C_{k}S_{k}^{-1}`

    :math:`m_{k}=m_{k}^{-} + K_{k}(y_{k}-\mu_{k})`

    :math:`P_{k}=P_{k}^{-} -K_{k}S_{k}K_{k}^{T}`
