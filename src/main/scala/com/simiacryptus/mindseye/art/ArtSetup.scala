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

import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.notebook.{MarkdownNotebookOutput, NotebookOutput}
import com.simiacryptus.sparkbook.{InteractiveSetup, RepeatedInteractiveSetup}

trait ArtSetup[T <: AnyRef] extends InteractiveSetup[T] {
  def cudaLog = false

  def maxImageSize = 10000

  val label = "Demo"

  override def apply(log: NotebookOutput): T = {
    TestUtil.addGlobalHandlers(log.getHttpd)
    VisionPipelineUtil.cudaReports(log, cudaLog)
    log.asInstanceOf[MarkdownNotebookOutput].setMaxImageSize(maxImageSize)
    super.apply(log)
  }
}

trait RepeatedArtSetup[T <: AnyRef] extends RepeatedInteractiveSetup[T] {
  def cudaLog = false

  def maxImageSize = 10000

  val label = "Demo"

  override def apply(log: NotebookOutput): T = {
    TestUtil.addGlobalHandlers(log.getHttpd)
    VisionPipelineUtil.cudaReports(log, cudaLog)
    log.asInstanceOf[MarkdownNotebookOutput].setMaxImageSize(maxImageSize)
    super.apply(log)
  }
}
