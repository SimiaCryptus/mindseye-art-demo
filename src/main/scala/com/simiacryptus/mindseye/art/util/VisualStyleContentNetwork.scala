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
import com.simiacryptus.mindseye.layers.java.{AssertDimensionsLayer, SumInputsLayer}
import com.simiacryptus.mindseye.network.PipelineNetwork

object VisualStyleContentNetwork {

}


case class VisualStyleContentNetwork
(
  styleLayers: Seq[VisionPipelineLayer],
  styleModifiers: Seq[VisualModifier],
  contentLayers: Seq[VisionPipelineLayer],
  contentModifiers: Seq[VisualModifier] = List(new ContentMatcher),
  styleUrl: Seq[String],
  precision: Precision = Precision.Float,
  viewLayer: Seq[Int] => Layer = _ => new PipelineNetwork(1),
  override val tileSize: Int = 400,
  override val tilePadding: Int = 16,
  override val minWidth: Int = 1,
  override val maxWidth: Int = 2048,
  override val maxPixels: Double = 5e6,
  override val magnification: Double = 1.0
) extends ImageSource(styleUrl) with VisualNetwork {

  def apply(canvas: Tensor, content: Tensor): Trainable = {
    val loadedImages = loadImages(VisualStyleNetwork.pixels(canvas))
    val styleModifier = styleModifiers.reduce(_ combine _)
    val contentModifier = contentModifiers.reduce(_ combine _)
    val grouped: Map[String, PipelineNetwork] = ((contentLayers.map(_.getPipelineName -> null) ++ styleLayers.groupBy(_.getPipelineName).toList).groupBy(_._1).mapValues(pipelineLayers => {
      val layers = pipelineLayers.flatMap(x => Option(x._2).toList.flatten)
      if (layers.isEmpty) null
      else SumInputsLayer.combine(layers.map(styleLayer => {
        val network = styleModifier.build(styleLayer, loadedImages.map(_.addRef()): _*)
        network.wrap(new AssertDimensionsLayer(1).setName(s"$styleModifier - $styleLayer"))
        network
      }): _*)
    })).toArray.map(identity).toMap
    loadedImages.foreach(_.freeRef())
    new SumTrainable(grouped.map(t => {
      val (name, styleNetwork) = t
      new TiledTrainable(canvas, viewLayer(canvas.getDimensions()), tileSize, tilePadding, precision) {
        override protected def getNetwork(regionSelector: Layer): PipelineNetwork = {
          val selection = regionSelector.eval(content).getDataAndFree.getAndFree(0)
          regionSelector.freeRef()
          MultiPrecision.setPrecision(SumInputsLayer.combine({
            Option(styleNetwork).map(_.addRef()).toList ++ contentLayers.filter(x => x.getPipelineName == name)
              .map(contentLayer => {
                val network = contentModifier.build(contentLayer, selection)
                network.wrap(new AssertDimensionsLayer(1).setName(s"$contentModifier - $contentLayer"))
                network
              })
          }: _*
          ), precision)
        }

        override protected def _free(): Unit = {
          if (null != styleNetwork) styleNetwork.freeRef()
          super._free()
        }
      }
    }).toArray: _*)
  }

}
