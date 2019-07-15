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

import java.lang
import java.util.concurrent.TimeUnit

import com.simiacryptus.mindseye.art.util.ArtUtil.withTrainingMonitor
import com.simiacryptus.mindseye.eval.Trainable
import com.simiacryptus.mindseye.lang.{Layer, Tensor}
import com.simiacryptus.mindseye.opt.line.{ArmijoWolfeSearch, LineSearchStrategy}
import com.simiacryptus.mindseye.opt.orient.{LBFGS, TrustRegionStrategy}
import com.simiacryptus.mindseye.opt.region.{CompoundRegion, RangeConstraint}
import com.simiacryptus.mindseye.opt.{IterativeTrainer, Step, TrainingMonitor}
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredJpg
import com.simiacryptus.sparkbook.util.Java8Util._

trait BasicOptimizer {

  def optimize(canvasImage: Tensor, trainable: Trainable)(implicit log: NotebookOutput) = {
    try {
      withMonitoredJpg(canvasImage.toRgbImage) {
        withTrainingMonitor(trainingMonitor => {
          log.eval(() => {
            val lineSearchInstance: LineSearchStrategy = lineSearchFactory
            IterativeTrainer.wrap(trainable)
              .setOrientation(orientation())
              .setMonitor(new TrainingMonitor() {
                override def clear(): Unit = trainingMonitor.clear()

                override def log(msg: String): Unit = trainingMonitor.log(msg)

                override def onStepFail(currentPoint: Step): Boolean = {
                  BasicOptimizer.this.onStepFail(trainable, currentPoint)
                }

                override def onStepComplete(currentPoint: Step): Unit = {
                  BasicOptimizer.this.onStepComplete(trainable, currentPoint)
                  trainingMonitor.onStepComplete(currentPoint)
                  super.onStepComplete(currentPoint)
                }
              })
              .setTimeout(trainingMinutes, TimeUnit.MINUTES)
              .setMaxIterations(trainingIterations)
              .setLineSearchFactory((_: CharSequence) => lineSearchInstance)
              .setTerminateThreshold(java.lang.Double.NEGATIVE_INFINITY)
              .runAndFree
              .asInstanceOf[lang.Double]
          })
        })
      }
    } finally {
      onComplete()
    }
  }

  def onComplete()(implicit log: NotebookOutput) = {}

  def onStepComplete(trainable: Trainable, currentPoint: Step) = {}

  def onStepFail(trainable: Trainable, currentPoint: Step) = {
    false
  }

  def trainingMinutes: Int = 60

  def trainingIterations: Int = 20

  def lineSearchFactory: LineSearchStrategy = new ArmijoWolfeSearch().setMaxAlpha(maxRate).setMinAlpha(1e-10).setAlpha(1).setRelativeTolerance(1e-5)

  def maxRate = 1e9

  def orientation() = {
    new TrustRegionStrategy(new LBFGS) {
      override def getRegionPolicy(layer: Layer) = trustRegion(layer)
    }
  }

  def trustRegion(layer: Layer) = {
    new CompoundRegion(
      //                new RangeConstraint().setMin(0).setMax(256),
      //                new FixedMagnitudeConstraint(canvasImage.coordStream(true)
      //                  .collect(Collectors.toList()).asScala
      //                  .groupBy(_.getCoords()(2)).values
      //                  .toArray.map(_.map(_.getIndex).toArray): _*),
      new RangeConstraint().setMin(0).setMax(256)
    )
  }
}
