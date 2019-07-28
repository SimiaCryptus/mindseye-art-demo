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

package com.simiacryptus.mindseye.art

import com.simiacryptus.aws.exe.EC2NodeSettings
import com.simiacryptus.mindseye.art.models.VGG19
import com.simiacryptus.mindseye.art.ops._
import com.simiacryptus.mindseye.art.util.ArtUtil._
import com.simiacryptus.mindseye.art.util._
import com.simiacryptus.mindseye.lang.Tensor
import com.simiacryptus.mindseye.opt.TrainingMonitor
import com.simiacryptus.notebook.{NotebookOutput, TableOutput}
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.LocalRunner

import scala.collection.JavaConverters._

object OperatorStatsSurvey_EC2 extends OperatorStatsSurvey with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def inputTimeoutSeconds = 600

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> "local[4]"
  )

}

object OperatorStatsSurvey_Local extends OperatorStatsSurvey with LocalRunner[Object] with NotebookRunner[Object] {
  override def inputTimeoutSeconds = 5

  override def s3bucket: String = ""
}

abstract class OperatorStatsSurvey extends ArtSetup[Object] with BasicOptimizer {

  val styleMagnification = 2.0
  val styleMin = 64
  val styleMax = 1280
  val stylePixelMax = 1e7
  val styles = List(
    "dmitry-levitzky",
    "edgar-degas",
    "edouard-manet"
  ).map(styleKey => styleKey -> findFiles(styleKey)).toMap
  val canvasSamples = 1
  val imageSize = 200

  override def cudaLog = false


  override def postConfigure(log: NotebookOutput) = {
    val tableOutput = new TableOutput()
    NotebookRunner.withMonitoredHtml(() => tableOutput.toHtmlTable) {
      for ((styleKey, styleFiles) <- styles) {
        for (styleLayer <- VGG19.values()) {
          for (op <- List(
            new ChannelMeanMatcher(),
            new GramMatrixMatcher().setTileSize(300),
            new GramMatrixEnhancer().setTileSize(300)
          )) {
            for (canvasName <- styles.values.flatMap(_.take(canvasSamples))) {
              tableOutput.putRow(Map(
                "style" -> styleKey,
                "image" -> canvasName,
                "layer" -> styleLayer.name(),
                "op" -> op.getClass.getSimpleName,
                "result" -> new VisualStyleNetwork(
                  styleLayers = List(styleLayer),
                  styleModifiers = List(op),
                  styleUrl = styleFiles,
                  magnification = OperatorStatsSurvey.this.styleMagnification,
                  minWidth = OperatorStatsSurvey.this.styleMin,
                  maxWidth = OperatorStatsSurvey.this.styleMax,
                  maxPixels = OperatorStatsSurvey.this.stylePixelMax
                ).apply(Tensor.fromRGB(VisionPipelineUtil.load(canvasName, imageSize)))
                  .measure(new TrainingMonitor)
                  .sum
              ).asJava)
            }
          }
        }
      }
    }(log)
    log.p(tableOutput.toHtmlTable)
    null
  }

}

