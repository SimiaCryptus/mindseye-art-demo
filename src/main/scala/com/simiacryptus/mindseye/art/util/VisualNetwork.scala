/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.art.util

import com.simiacryptus.lang.ref.ReferenceCountingBase
import com.simiacryptus.mindseye.art.SumTrainable
import com.simiacryptus.mindseye.eval.Trainable
import com.simiacryptus.mindseye.lang.cudnn.Precision
import com.simiacryptus.mindseye.lang.{Layer, PointSample, Tensor}
import com.simiacryptus.mindseye.opt.TrainingMonitor

trait VisualNetwork {
  def precision: Precision

  def apply(canvas: Tensor, content: Tensor): Trainable

  def +(value: VisualNetwork): VisualNetwork = {
    val inner = this
    new VisualNetwork {
      require(inner.precision == value.precision)

      override def precision: Precision = inner.precision

      override def apply(canvas: Tensor, content: Tensor): Trainable = new SumTrainable(
        inner.apply(canvas, content),
        value.apply(canvas, content)
      )
    }
  }


  def *(value: Double): VisualNetwork = {
    val inner = this
    new VisualNetwork {
      override def precision: Precision = inner.precision

      override def apply(canvas: Tensor, content: Tensor): Trainable = new ReferenceCountingBase with Trainable {
        lazy val innerTrainable = inner.apply(canvas, content)

        override def measure(monitor: TrainingMonitor): PointSample = {
          val pointSample = innerTrainable.measure(monitor)
          val scaled = new PointSample(
            pointSample.delta.scale(value),
            pointSample.weights.addRef(),
            pointSample.sum * value,
            pointSample.rate,
            pointSample.count
          )
          pointSample.freeRef()
          scaled
        }

        override def getLayer: Layer = {
          innerTrainable.getLayer
        }
      }
    }
  }
}
