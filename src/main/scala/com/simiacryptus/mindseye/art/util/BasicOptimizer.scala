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

import java.awt.image.BufferedImage
import java.lang
import java.util.UUID
import java.util.concurrent.TimeUnit

import com.simiacryptus.aws.S3Util
import com.simiacryptus.mindseye.art.util.ArtUtil.{setPrecision, withTrainingMonitor}
import com.simiacryptus.mindseye.eval.Trainable
import com.simiacryptus.mindseye.lang.cudnn.Precision
import com.simiacryptus.mindseye.lang.{Layer, Tensor}
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.opt.line.{ArmijoWolfeSearch, LineSearchStrategy}
import com.simiacryptus.mindseye.opt.orient.{LBFGS, TrustRegionStrategy}
import com.simiacryptus.mindseye.opt.region.{RangeConstraint, TrustRegion}
import com.simiacryptus.mindseye.opt.{IterativeTrainer, Step, TrainingMonitor}
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.NotebookRunner
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredJpg
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.Logging

import scala.collection.mutable.ArrayBuffer

trait BasicOptimizer extends Logging {

  def optimize(canvasImage: Tensor, trainable: Trainable)(implicit log: NotebookOutput) = {
    try {
      def currentImage = renderingNetwork(canvasImage.getDimensions).eval(canvasImage).getDataAndFree.getAndFree(0).toRgbImage

      val timelineAnimation = new ArrayBuffer[BufferedImage]()
      withMonitoredJpg(() => currentImage) {
        log.subreport("Optimization_" + UUID.randomUUID().toString, (sub: NotebookOutput) => {
          NotebookRunner.withMonitoredGif(() => timelineAnimation.toList ++ List(currentImage)) {
            withTrainingMonitor(trainingMonitor => {
              sub.eval(() => {
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
                      if (0 < logEvery && 0 == currentPoint.iteration % logEvery) {
                        val image = currentImage
                        timelineAnimation += image
                        sub.p(sub.jpg(image, "Iteration " + currentPoint.iteration))
                      }
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
              null
            })(sub)
          }(sub)
          null
        })
      }
    } finally {
      try {
        onComplete()
      } catch {
        case e: Throwable => logger.warn("Error running onComplete", e)
      }
    }
  }

  def renderingNetwork(dims: Seq[Int]): PipelineNetwork = new PipelineNetwork(1)

  def onStepComplete(trainable: Trainable, currentPoint: Step) = {
    setPrecision(trainable, Precision.Float)
  }

  def onStepFail(trainable: Trainable, currentPoint: Step): Boolean = {
    setPrecision(trainable, Precision.Double)
  }

  def onComplete()(implicit log: NotebookOutput): Unit = {
    S3Util.upload(log)
  }

  def logEvery = 5

  def trainingMinutes: Int = 60

  def trainingIterations: Int = 20

  def lineSearchFactory: LineSearchStrategy = new ArmijoWolfeSearch().setMaxAlpha(maxRate).setMinAlpha(1e-10).setAlpha(1).setRelativeTolerance(1e-5)

  def maxRate = 1e9

  def orientation() = {
    new TrustRegionStrategy(new LBFGS) {
      override def getRegionPolicy(layer: Layer) = trustRegion(layer)
    }
  }

  def trustRegion(layer: Layer): TrustRegion = {
    //    new CompoundRegion(
    //      //                new RangeConstraint().setMin(0).setMax(256),
    //      //                new FixedMagnitudeConstraint(canvasImage.coordStream(true)
    //      //                  .collect(Collectors.toList()).asScala
    //      //                  .groupBy(_.getCoords()(2)).values
    //      //                  .toArray.map(_.map(_.getIndex).toArray): _*),
    //    )
    new RangeConstraint().setMin(0).setMax(256)
  }
}
