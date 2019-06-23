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

import com.simiacryptus.mindseye.art._
import com.simiacryptus.mindseye.art.ops.ContentMatcher
import com.simiacryptus.mindseye.eval.Trainable
import com.simiacryptus.mindseye.lang.cudnn.{MultiPrecision, Precision}
import com.simiacryptus.mindseye.lang.{Layer, Tensor}
import com.simiacryptus.mindseye.layers.java.SumInputsLayer
import com.simiacryptus.mindseye.network.PipelineNetwork

object CartesianStyleContentNetwork {

}

case class CartesianStyleContentNetwork
(
  styleLayers: Seq[VisionPipelineLayer],
  styleModifiers: Seq[VisualModifier],
  contentLayers: Seq[VisionPipelineLayer],
  contentModifiers: Seq[VisualModifier] = List(new ContentMatcher),
  styleUrl: Seq[String],
  precision: Precision = Precision.Float,
  viewLayer: Layer = new PipelineNetwork(1),
  override val tileSize: Int = 400,
  override val tilePadding: Int = 16,
  override val minWidth: Int = 1,
  override val maxWidth: Int = 2048,
  override val maxPixels: Double = 5e6,
  override val magnification: Double = 1.0
) extends ImageSource(styleUrl) {

  def apply(canvas: Tensor, content: Tensor): Trainable = {
    val loadedImages = loadImages(CartesianStyleNetwork.pixels(canvas))
    val grouped: Map[String, PipelineNetwork] = contentLayers.map(_.getPipeline.name -> null).toMap ++ styleLayers.groupBy(_.getPipeline.name).mapValues(pipelineLayers => {
      SumInputsLayer.combine(pipelineLayers.filter(x => styleLayers.contains(x)).map(styleModifiers.reduce(_ combine _).build(_, loadedImages: _*)): _*)
    })
    new SumTrainable(grouped.map(t => {
      val (name, styleNetwork) = t
      new TiledTrainable(canvas, viewLayer, tileSize, tilePadding, precision) {
        override protected def getNetwork(regionSelector: Layer): PipelineNetwork = {
          val network = if (null == styleNetwork) {
            MultiPrecision.setPrecision(SumInputsLayer.combine(
              contentLayers.filter(x => x.getPipeline.name == name)
                .map(layer => contentModifiers.reduce(_ combine _).build(
                  layer,
                  regionSelector.eval(content).getDataAndFree.getAndFree(0)
                )): _*
            ), precision).asInstanceOf[PipelineNetwork]
          } else {
            MultiPrecision.setPrecision(SumInputsLayer.combine((
              List(styleNetwork.addRef()) ++ contentLayers.filter(x => x.getPipeline.name == name)
                .map(layer => contentModifiers.reduce(_ combine _).build(
                  layer,
                  regionSelector.eval(content).getDataAndFree.getAndFree(0)
                ))
              ): _*), precision).asInstanceOf[PipelineNetwork]
          }
          regionSelector.freeRef()
          network
        }
      }
    }).toList: _*)
  }

}
