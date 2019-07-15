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

package com.simiacryptus.mindseye.art.index

import java.awt.Desktop
import java.io.File
import java.net.{InetAddress, URI}
import java.util.UUID
import java.util.concurrent.TimeUnit

import com.fasterxml.jackson.annotation.JsonIgnore
import com.simiacryptus.aws.exe.EC2NodeSettings
import com.simiacryptus.mindseye.art._
import com.simiacryptus.mindseye.art.index.BooleanIterator._
import com.simiacryptus.mindseye.art.models.VGG19._
import com.simiacryptus.mindseye.art.ops.{ChannelMeanMatcher, GramMatrixMatcher}
import com.simiacryptus.mindseye.art.util.ArtUtil._
import com.simiacryptus.mindseye.art.util._
import com.simiacryptus.mindseye.eval.ArrayTrainable
import com.simiacryptus.mindseye.lang.cudnn.{MultiPrecision, Precision}
import com.simiacryptus.mindseye.lang.{Layer, Tensor}
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer.PoolingMode
import com.simiacryptus.mindseye.layers.cudnn.{BandAvgReducerLayer, BandReducerLayer, GramianLayer}
import com.simiacryptus.mindseye.layers.java._
import com.simiacryptus.mindseye.network.{PipelineNetwork, SimpleLossNetwork}
import com.simiacryptus.mindseye.opt.IterativeTrainer
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch
import com.simiacryptus.mindseye.opt.orient.OwlQn
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.notebook.{FormQuery, MarkdownNotebookOutput, NotebookOutput}
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.{LocalRunner, ScalaJson}
import com.simiacryptus.sparkbook.{AWSNotebookRunner, EC2Runner, NotebookRunner}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.concurrent.duration._
import scala.concurrent.{Await, Future}
import scala.util.{Random, Try}

object BooleanIterator_EC2 extends BooleanIterator with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def urlBase: String = String.format("http://%s:1080/etc/", InetAddress.getLocalHost.getHostAddress)

  override def inputTimeoutSeconds = 600

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> spark_master
  )

  override def spark_master = "local[1]"

}

object BooleanIterator_Local extends BooleanIterator with LocalRunner[Object] with NotebookRunner[Object] {
  override val urlBase: String = "http://localhost:1080/etc/"

  override def inputTimeoutSeconds = 30

  override def spark_master = "local[1]"

}


abstract class BooleanIterator extends ArtSetup[Object] with BasicOptimizer {

  val archiveUrl = "file:///C:/Users/andre/data/images/"
  val toDisplay = 1000
  val indexResolution = 512
  val thumbnailResolution = 128
  val positiveSeeds: Array[String] = "self-portrait-1982(2).jpg!Large.jpg\nleonaert-bramer-dying-vision-of-maria-magdalen-1.jpg!Large.jpg\nfayum-04.jpg!Large.jpg\nplot-1.jpg!Large.jpg\nnude.jpg!Large.jpg\nportrait-of-the-artist-1879.jpg!Large.jpg\nhomer-epic-poetry.jpg!Large.jpg\nbalinese-ceremony.jpg!Large.jpg\nserious-art-1986.jpg!Large.jpg\nyellow-bluebells-1970.jpg!Large.jpg\nportrait-of-a-man-in-a-bow-tie-1890.jpg!Large.jpg\nat-aphrodite-s-cradle-1908.jpg!Large.jpg\nportrait-1943.jpg!Large.jpg\nberenice-1640.jpg!Large.jpg\njohannes-moreelse-heraclitus-google-art-project-1.jpg!Large.jpg\nvenere-anadiomene-ii-1944.jpg!Large.jpg\nportrait-de-madame-dot-zac-1902.jpg!Large.jpg\namerican-scene.jpg!Large.jpg\nnude-in-a-blue-interior.jpg!Large.jpg\npoets-in-my-house1953.jpg!Large.jpg\njapanese-woman-1879.jpg!Large.jpg\nmrs-william-theobald-n-e-sarah-cooke-1832.jpg!Large.jpg\ntatar-interior-1933.jpg!Large.jpg\na-youth-with-a-jug.jpg!Large.jpg".split("\n").toList.toArray
  val negativeSeeds: Array[String] = "la-quinta-trompeta.JPG!Large.JPG\nsamson-and-delilah-1668.jpg!Large.jpg\njolly-toper-1629.jpg!Large.jpg\naltarpiece-with-the-martyrdom-of-st-sebastian-1507.jpg!Large.jpg\ngarden-hose.jpg!Large.jpg\na-dangerous-game.jpg!Large.jpg\nmerry-family-1668.jpg!Large.jpg\ntzar-ivan-the-terrible-asks-abbot-cornelius-to-mow-him-to-the-monks.jpg!Large.jpg\ndisplay-image-2.jpg!Large.jpg\ntristram-and-isolde-1916.jpg!Large.jpg\nthe-tooth-extractor.jpg!Large.jpg\nmadonna-and-child-enthroned-with-two-angels-1480.jpg!Large.jpg\nportrait-of-a-knight-1510.jpg!Large.jpg\ndanae-1531(2).jpg!Large.jpg\nmankind-s-struggle-for-lasting-peace-detail-1953.jpg!Large.jpg\nthe-corner-of-the-table-1872.jpg!Large.jpg\nchaucer-at-the-court-of-edward-iii-1851.jpg!Large.jpg\n11223787-638686919599919-7104767734246742869-o.jpg!Large.jpg\ndie-liebenden-1744.jpg!Large.jpg\nsundblom-634-copy-l.jpg!Large.jpg\nmidsummer-1887.jpg!Large.jpg\napollo-1874(1).jpg!Large.jpg\nlamentation-on-the-dead-christ-1566.jpg!Large.jpg\nthe-family-of-the-artist-1635.jpg!Large.jpg".split("\n").toList.toArray
  val content_url = "file:///C:/Users/andre/Downloads/IMG_20181031_164826422.jpg"
  val initialSamples = 20
  val incrementalSamples = 20
  val sampleEpochs = 0
  val hiddenLayer1 = 128
  val dropoutSamples = 5
  val dropoutFactor = Math.pow(0.5, 0.5)
  val classificationPaintingBias = 0.5
  val signalMatchBias = 0.1

  @JsonIgnore def spark_master: String

  def urlBase: String

  override def cudaLog = false

  override def postConfigure(log: NotebookOutput) = {
    implicit val sparkSession = sparkFactory
    implicit val exeCtx = scala.concurrent.ExecutionContext.Implicits.global

    var precision = Precision.Double
    val visionLayer = VGG19_1e1
    val dim = visionLayer.getOutputChannels

    val index = sparkSession.read.parquet(archiveUrl).cache()
    index.printSchema()

    def findRows(example: String*): Dataset[Row] = {
      index.where(example.map(index("file").contains(_)).reduceOption(_ or _).getOrElse(lit(false)).and(
        index("layer").eqNullSafe(lit(visionLayer.name()))
      ).and(
        index("resolution").eqNullSafe(lit(indexResolution))
      ))
    }

    val positiveExamples = findRows(positiveSeeds.map(_.trim).filter(!_.isEmpty): _*).collect().toBuffer
    val negativeExamples = findRows(negativeSeeds.map(_.trim).filter(!_.isEmpty): _*).collect().toBuffer

    def avoid = positiveExamples.union(negativeExamples).map(_.getAs[String]("file")).distinct.toArray

    index.groupBy("layer", "resolution").agg(count(index("file")).as("count")).foreach(row => println(row.toString))

    def filterIndex = index.where(
      index("layer").eqNullSafe(lit(visionLayer.name())).and(
        !index("file").isin(avoid: _*)
      ).and(
        index("resolution").eqNullSafe(lit(indexResolution))
      )
    )

    require(!filterIndex.isEmpty)
    val pngCache = new mutable.HashMap[String, File]()
    val (meanSignalPreview: Tensor, covarianceSignalPreview: Tensor) = stats(filterIndex.limit(100))
    val innerClassifier = PipelineNetwork.wrap(1,
      new LinearActivationLayer().setScale(1e-2 * Math.pow(meanSignalPreview.rms(), -1)).freeze(),
      new FullyConnectedLayer(Array(1, 1, dim), Array(hiddenLayer1)),
      new BiasLayer(hiddenLayer1),
      //      new ReLuActivationLayer(),
      new SigmoidActivationLayer(),
      new DropoutNoiseLayer(dropoutFactor),
      new FullyConnectedLayer(Array(hiddenLayer1), Array(2)),
      new BiasLayer(2),
      new SoftmaxLayer()
    )
    var classifier: Layer = innerClassifier
    classifier = new StochasticSamplingSubnetLayer(classifier, dropoutSamples)
    val selfEntropyNet = new PipelineNetwork(1)
    selfEntropyNet.wrap(classifier)
    selfEntropyNet.wrap(new EntropyLossLayer(), selfEntropyNet.getHead, selfEntropyNet.getHead)

    def bestSamples(sample: Int) = sparkSession.createDataFrame(sparkSession.sparkContext.parallelize(filterIndex.rdd.sortBy(row => {
      val array = row.getAs[Seq[Double]]("data").toArray
      val tensor = new Tensor(array, 1, 1, array.length)
      val result = selfEntropyNet.eval(tensor).getDataAndFree.getAndFree(0)
      val v = result.get(0)
      tensor.freeRef()
      result.freeRef()
      -v
    }).take(sample)), index.schema)

    def newConfirmationBatch(index: DataFrame, sample: Int, log: NotebookOutput) = {
      val seed = index.rdd.map(_.getAs[String]("file")).distinct().take(sample).map(_ -> true).toMap
      val ids = seed.mapValues(_ => UUID.randomUUID().toString).toArray.toMap
      Await.result(Future.sequence(for ((k, v) <- seed) yield Future {
        try {
          val filename = k.split('/').last.toLowerCase.stripSuffix(".png") + ".png"
          pngCache.getOrElseUpdate(k, log.pngFile(VisionPipelineUtil.load(k, 256), new File(log.getResourceDir, filename)))
        } catch {
          case e: Throwable => e.printStackTrace()
        }
      }), 10 minutes)
      new FormQuery[Map[String, Boolean]](log.asInstanceOf[MarkdownNotebookOutput]) {

        override def height(): Int = 800

        override protected def getDisplayHtml: String = ""

        override protected def getFormInnerHtml: String = {
          (for ((k, v) <- getValue) yield {
            val filename = k.split('/').last.toLowerCase.stripSuffix(".png") + ".png"
            pngCache.getOrElseUpdate(k, log.pngFile(VisionPipelineUtil.load(k, 256), new File(log.getResourceDir, filename)))
            s"""<input type="checkbox" name="${ids(k)}" value="true"><img src="etc/$filename"><br/>"""
          }).mkString("\n")
        }

        override def valueFromParams(parms: java.util.Map[String, String]): Map[String, Boolean] = {
          (for ((k, v) <- getValue) yield {
            k -> parms.getOrDefault(ids(k), "false").toBoolean
          })
        }
      }.setValue(seed).print().get(6000, TimeUnit.SECONDS)
    }

    def trainEpoch(log: NotebookOutput) = {
      withTrainingMonitor(monitor => {
        classifier.asInstanceOf[StochasticSamplingSubnetLayer].clearNoise
        log.eval(() => {
          val search = new ArmijoWolfeSearch
          IterativeTrainer.wrap(new ArrayTrainable((positiveExamples.map(x => Array(
            new Tensor(x.getAs[Seq[Double]]("data").toArray, 1, 1, dim),
            new Tensor(Array(1.0, 0.0), 1, 1, 2)
          )).toList ++ negativeExamples.map(x => Array(
            new Tensor(x.getAs[Seq[Double]]("data").toArray, 1, 1, dim),
            new Tensor(Array(0.0, 1.0), 1, 1, 2)
          )).toList).toArray, new SimpleLossNetwork(classifier, new EntropyLossLayer())))
            .setMaxIterations(100)
            .setIterationsPerSample(5)
            .setLineSearchFactory((n: CharSequence) => search)
            .setOrientation(new OwlQn())
            .setMonitor(monitor)
            .runAndFree().toString
        })
        classifier.asInstanceOf[StochasticSamplingSubnetLayer].clearNoise
        null
      })(log)
    }

    if (positiveExamples.isEmpty || negativeExamples.isEmpty) {
      // Build Tag Model
      val (newPositives, newNegatives) = newConfirmationBatch(index = filterIndex, sample = initialSamples, log = log).partition(_._2)
      positiveExamples ++= newPositives.keys.map(findRows(_).head()).toList
      negativeExamples ++= newNegatives.keys.map(findRows(_).head()).toList
    }
    trainEpoch(log = log)

    for (i <- 0 until sampleEpochs) {
      val (newPositives, newNegatives) = newConfirmationBatch(index = bestSamples(incrementalSamples), sample = incrementalSamples, log = log).partition(_._2)
      positiveExamples ++= newPositives.keys.map(findRows(_).head()).toList
      negativeExamples ++= newNegatives.keys.map(findRows(_).head()).toList
      trainEpoch(log = log)
    }

    // Write projector
    val dataframe = sparkSession.createDataFrame(filterIndex.rdd.map(r => {
      val array = r.getAs[Seq[Double]]("data").toArray
      val tensor = new Tensor(array, 1, 1, array.length)
      val result = classifier.eval(tensor).getDataAndFree.getAndFree(0)
      val v = result.get(0)
      tensor.freeRef()
      result.freeRef()
      r -> v
    })
      //.filter(_._2 > .5)
      .sortBy(-_._2)
      .map(_._1), index.schema).limit(toDisplay)

    log.subreport("Projector", (sub: NotebookOutput) => {
      val fileKeys = dataframe.select("file").rdd.map(_.getString(0)).distinct().collect()
      val pngFile: String = getThumbnailImage(fileKeys, thumbnailResolution)(sub)
      val embeddingConfigs = for (Row(pipeline: String, resolution: Int, layer: String) <- dataframe.select("pipeline", "resolution", "layer").distinct().collect()) yield {
        val label = s"${pipeline}_${layer}_$resolution"
        sub.h1(label)
        val embeddings = dataframe.where((dataframe("pipeline") eqNullSafe pipeline) and (dataframe("resolution") eqNullSafe resolution) and (dataframe("layer") eqNullSafe layer))
          .select("file", "data").limit(toDisplay).collect()
          .map({ case Row(file: String, data: Seq[Double]) => file -> data.toArray }).toMap
          .mapValues(data => new Tensor(data, 1, 1, data.length))

        val config = getProjectorConfig(label, embeddings, urlBase, thumbnailResolution, fileKeys, pngFile)(sub)
        logEmbedding(urlBase, config)(sub)
        config
      }
      displayEmbedding(urlBase, embeddingConfigs: _*)(sub)
    })


    paint(dataframe, precision, visionLayer, innerClassifier, log, "Painting_Boolean")
    null
  }

  @JsonIgnore def sparkFactory: SparkSession = {
    val builder = SparkSession.builder()
    import scala.collection.JavaConverters._
    VisionPipelineUtil.getHadoopConfig().asScala.foreach(t => builder.config(t.getKey, t.getValue))
    builder.master("local[8]").getOrCreate()
  }

  def paint(dataframe: Dataset[Row], p: Precision, visionLayer: VisionPipelineLayer, innerClassifier: Layer, log: NotebookOutput, title: String) = {
    var precision = p
    val (meanSignal: Tensor, covarianceSignal: Tensor) = stats(dataframe)

    def train(canvas: Tensor, log: NotebookOutput): Tensor = {
      train2(train1(canvas, log), log)
    }

    def trainNetwork(canvas: Tensor, network: PipelineNetwork, log: NotebookOutput) = {
      network.freeze()
      new BasicOptimizer {
        override def maxRate: Double = 1e10

        override def trainingIterations: Int = 50
      }.optimize(canvas, new TiledTrainable(canvas, 450, 16, precision) {
        override protected def getNetwork(regionSelector: Layer): PipelineNetwork = {
          regionSelector.freeRef()
          MultiPrecision.setPrecision(network.addRef(), precision).asInstanceOf[PipelineNetwork]
        }
      })(log)
    }

    def train1(canvas: Tensor, log: NotebookOutput): Tensor = {
      val visionNetwork = visionLayer.getNetwork
      var network = visionNetwork.copyPipeline()
      network.wrap(new EntropyLossLayer(),
        network.wrap(innerClassifier,
          network.wrap(new BandReducerLayer().setMode(PoolingMode.Avg))),
        network.constValue(new Tensor(Array(1.0, 0.0), 1, 1, 2))
      )
      network.wrap(new LinearActivationLayer().setBias(classificationPaintingBias))
      //        network.wrap(new NthPowerActivationLayer().setPower(-1))
      network = PipelineNetwork.combine(
        new ProductInputsLayer(),
        PipelineNetwork.combine(
          new SumInputsLayer(),
          //          new GramMatrixMatcher().buildWithModel(visionNetwork, covarianceSignal.addRef()),
          new ChannelMeanMatcher().buildWithModel(visionNetwork, meanSignal)
        ).andThenWrap(new LinearActivationLayer().setBias(signalMatchBias)),
        network
      )
      trainNetwork(canvas, network, log)
      canvas
    }

    def train2(canvas: Tensor, log: NotebookOutput): Tensor = {
      val visionNetwork = visionLayer.getNetwork
      var network = visionNetwork.copyPipeline()
      network.wrap(new EntropyLossLayer(),
        network.wrap(innerClassifier,
          network.wrap(new BandReducerLayer().setMode(PoolingMode.Avg))),
        network.constValue(new Tensor(Array(1.0, 0.0), 1, 1, 2))
      )
      network.wrap(new LinearActivationLayer().setBias(classificationPaintingBias))
      //        network.wrap(new NthPowerActivationLayer().setPower(-1))
      network = PipelineNetwork.combine(
        new ProductInputsLayer(),
        PipelineNetwork.combine(
          new SumInputsLayer(),
          new GramMatrixMatcher().buildWithModel(visionNetwork, covarianceSignal.addRef()),
          new ChannelMeanMatcher().buildWithModel(visionNetwork, meanSignal)
        ).andThenWrap(new LinearActivationLayer().setBias(signalMatchBias)),
        network
      )
      trainNetwork(canvas, network, log)
      canvas
    }

    log.subreport(title, (sub: NotebookOutput) => {
      new GeometricResolutionSequence {
        override def minResolution: Int = 512

        override def maxResolution: Int = 1200

        override def resolutionSteps: Int = 5
      }.resolutions.foldLeft[Option[Tensor]](None)((imgOpt, res) => imgOpt.map(img => {
        precision = Precision.Float
        train(Tensor.fromRGB(TestUtil.resize(img.toRgbImage, res, true)), log = sub)
      }).orElse({
        precision = Precision.Double
        val img = VisionPipelineUtil.load(content_url, res)
        Option(train(Tensor.fromRGB(img), log = sub))
      }))
    })
  }

  def stats(dataframe: Dataset[Row]) = {
    val pixels = dataframe.select("data").rdd.map(_.getAs[Seq[Double]](0).toArray)
    val dim = pixels.first().length
    val pixelCnt = pixels.count().toInt
    val meanSignal = new Tensor(pixels.reduce((a, b) => a.zip(b).map(t => t._1 + t._2)).map(_ / pixelCnt), 1, 1, dim)
    val dataIn = new Tensor(pixels.reduce(_ ++ _), 1, pixelCnt, dim)
    val covarianceSignal = new GramianLayer().eval(dataIn).getDataAndFree.getAndFree(0)
    dataIn.freeRef()
    (meanSignal, covarianceSignal)
  }

  def select(index: DataFrame, exampleRow: Row, window: Int)(implicit sparkSession: SparkSession): DataFrame = {
    val files = index.where(index("resolution").eqNullSafe(exampleRow.getAs[Int]("resolution")).and(
      index("pipeline").eqNullSafe(exampleRow.getAs[String]("pipeline"))
    ).and(
      index("layer").eqNullSafe(exampleRow.getAs[String]("layer"))
    )).rdd.sortBy(r => r.getAs[Seq[Double]]("data").zip(exampleRow.getAs[Seq[Double]]("data"))
      .map(t => t._1 - t._2).map(x => x * x).sum).map(_.getAs[String]("file")).distinct.take(window).toSet
    index.filter(r => files.contains(r.getAs[String]("file")))
  }
}

object BooleanIterator {

  def indexImages(visionPipeline: => VisionPipeline[VisionPipelineLayer], toIndex: Int, indexResolution: Int, archiveUrl: String)
                 (files: String*)
                 (implicit sparkSession: SparkSession): DataFrame = {
    val indexed = Try {
      val previousIndex = sparkSession.read.parquet(archiveUrl)
      previousIndex.select("file").rdd.map(_.getString(0)).distinct().collect().toSet
    }.getOrElse(Set.empty)
    val allFiles = Random.shuffle(files.filter(!indexed.contains(_)).toList).distinct.take(toIndex)
    if (allFiles.isEmpty) sparkSession.emptyDataFrame
    else index(visionPipeline, indexResolution, allFiles: _*)
  }

  def index(pipeline: => VisionPipeline[VisionPipelineLayer], imageSize: Int, images: String*)
           (implicit sparkSession: SparkSession) = {
    val rows = sparkSession.sparkContext.parallelize(images, images.length).flatMap(file => {
      val layers = pipeline.getLayers
      val canvas = Tensor.fromRGB(VisionPipelineUtil.load(file, imageSize))
      val tuples = layers.foldLeft(List(canvas))((input, layer) => {
        val l = layer.getLayer
        val tensors = input ++ List(l.eval(input.last).getDataAndFree.getAndFree(0))
        l.freeRef()
        tensors
      })
      tuples.head.freeRef()
      val reducerLayer = new BandAvgReducerLayer()
      val rows = (layers.map(_.name()) zip tuples.tail).toMap
        .mapValues(data => {
          val tensor = reducerLayer.eval(data).getDataAndFree.getAndFree(0)
          data.freeRef()
          val doubles = tensor.getData.clone()
          tensor.freeRef()
          doubles
        }).map(t => {
        Array(file, imageSize, pipeline.name, t._1, t._2)
      })
      reducerLayer.freeRef()
      println(s"Indexed $file")
      rows
    }).cache()
    sparkSession.createDataFrame(rows.map(Row(_: _*)), StructType(Array(
      StructField("file", StringType),
      StructField("resolution", IntegerType),
      StructField("pipeline", StringType),
      StructField("layer", StringType),
      StructField("data", ArrayType(DoubleType))
    )))
  }

  def getProjectorConfig(name: String, embeddings: Map[String, Tensor], urlBase: String, spriteSize: Int, images: Array[String], pngFile: String)
                        (implicit log: NotebookOutput) = {
    val id = UUID.randomUUID().toString
    log.p(log.file(images.map(embeddings).map(_.getData.map("%.4f".format(_)).mkString("\t")).mkString("\n"), s"$id.tensors.tsv", "Data"))
    log.p(log.file((
      List(List("Label", "URL").mkString("\t")) ++
        images.map(img => List(
          img.split('/').last,
          img.replaceAll("s3a://simiacryptus/", "https://simiacryptus.s3.amazonaws.com/")
            .replaceAll("s3a://", "https://s3.amazonaws.com/")
        ).mkString("\t"))
      ).mkString("\n"), s"$id.metadata.tsv", "Metadata"))
    Map(
      "tensorName" -> name,
      "tensorShape" -> Array(images.size, embeddings.head._2.length()),
      "tensorPath" -> s"$urlBase$id.tensors.tsv",
      "metadataPath" -> s"$urlBase$id.metadata.tsv",
      "sprite" -> Map(
        "imagePath" -> s"$urlBase$pngFile",
        "singleImageDim" -> Array(spriteSize, spriteSize)
      )
    )
  }

  def getThumbnailImage(images: Array[String], spriteSize: Int)
                       (implicit log: NotebookOutput) = {
    val rowCols = Math.sqrt(images.length).ceil.toInt
    val sprites = new ImgTileAssemblyLayer(rowCols, rowCols).eval(
      (images ++ images.take((rowCols * rowCols) - images.length)).par.map(VisionPipelineUtil.load(_, spriteSize, spriteSize)).map(Tensor.fromRGB(_)).toArray: _*
    ).getDataAndFree.getAndFree(0)
    val pngTxt = log.png(sprites.toRgbImage, "sprites")
    log.p(pngTxt)
    val pngFile = pngTxt.split('(').last.stripSuffix(")").stripPrefix("etc/")
    pngFile
  }


  def displayEmbedding(urlBase: String, embedding: Map[String, Any]*)
                      (implicit log: NotebookOutput) = {
    val projectorUrl: String = logEmbedding(urlBase, embedding: _*)
    Try {
      Desktop.getDesktop.browse(new URI(projectorUrl))
    }
    embedding
  }

  def logEmbedding(urlBase: String, embedding: Map[String, Any]*)
                  (implicit log: NotebookOutput) = {
    val id = UUID.randomUUID().toString
    log.p(log.file(ScalaJson.toJson(Map("embeddings" -> embedding.map(identity).toArray)), s"$id.json", "Projector Config"))
    val projectorUrl = s"""https://projector.tensorflow.org/?config=$urlBase$id.json"""
    log.p(s"""<a href="$projectorUrl">Projector</a>""")
    projectorUrl
  }
}