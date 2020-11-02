import scala.util.Random
import breeze.linalg._

class LinearRegression() {

  var weight: DenseVector[Double] = DenseVector.ones[Double](0)

  def predict(X: DenseMatrix[Double]): DenseVector[Double] = {
    X * weight
  }

  def mse(y: DenseVector[Double], y_preds: DenseVector[Double]): Double = {
    val diff = y.toArray zip y_preds.toArray map ( z => scala.math.pow(z._1 - z._2, 2))
    diff.sum / y.length.toDouble
  }

  def mae(y: DenseVector[Double], y_preds: DenseVector[Double]): Double = {
    val diff = y.toArray zip y_preds.toArray map ( z => scala.math.abs(z._1 - z._2))
    diff.sum / y.length.toDouble
  }

  def fit(X: DenseMatrix[Double], y: DenseVector[Double], lr: Double, iterations: Int): Unit = {
    weight = DenseVector.ones[Double](X.cols)
    var y_preds = DenseVector.zeros[Double](y.length)
    var loss_best = Double.MaxValue
    var loss = Double.MaxValue
    var gradients = DenseVector.zeros[Double](X.cols)

    for (i <- 0 to iterations; if loss_best >= loss) {
      loss_best = loss
      y_preds = predict(X)
      gradients = pinv(X) * (y_preds - y)
      weight -= lr * gradients
      y_preds = predict(X)
      loss = mse(y, y_preds)

      if ((i&0xff) == 0xff) {
        val loss_mae = mae(y, y_preds)
        println(s"iter: $i, mse: $loss, mae: $loss_mae, weight: $weight")
      }
    }
  }
  def get_weight(): DenseVector[Double] ={
    weight
  }

  def set_weight(w: DenseVector[Double]): Unit ={
    weight = w
  }

}

object mainspace {
  def main(args: Array[String]) {
    /*
    // test with real data
    val (x, y, test) = read_data()
    val model = new LinearRegression()
    val lr = 0.001
    val iterations = 30000
    model.fit(x(0 to 1599, 0 to 24), y(0 to 1599), lr, iterations)

    val test_preds = model.predict(x(1600 to 1999, 0 to 24))
    val weight = model.get_weight()

    println(s"\nprediction weight:")
    for (w <- weight){
      print(s"$w ")
    }
    println("\n\nPredictions:")
    var acc:Int = 0
    for (i <- 0 to test_preds.length - 1) {
      if (y(i + 1600)==scala.math.round(test_preds(i))){
        acc=acc+1
      }
      print(s"$acc of $i ")
      println(y(i + 1600), scala.math.round(test_preds(i)), test_preds(i))
    }
    */
    // test with generate data
    val true_weight = DenseVector[Double](1.0, -2.0, -3.0, 4.0, 5.0)
    val (x, y) = generate_lineardata(true_weight)
    val model = new LinearRegression()
    val lr = 0.001
    val iterations = 30000
    model.fit(x, y, lr, iterations)

    val y_preds = model.predict(x)
    val weight = model.get_weight()
    val mse = model.mse(y_preds, y)

    println("\ntrue weight:")
    for (w <- true_weight){
      print(s"$w ")
    }
    println(s"\nprediction weight:")
    for (w <- weight){
      print(s"$w ")
    }
    println(s"\nMSE: $mse")

    println("\n\n")
    for (i <- 0 to y_preds.length - 1){
      println(y(i), y_preds(i))
    }
  }


  def generate_lineardata(weight: DenseVector[Double], size: Int = 100): (DenseMatrix[Double], DenseVector[Double]) = {
    val X_max = 100
    val noise_max = 10
    val rand = new Random
    val X = DenseMatrix.fill[Double](size, weight.length)(rand.nextDouble() * X_max)
    val noise = DenseVector.fill[Double](size)(rand.nextDouble() * noise_max)
    val y = X * weight + noise
    (X, y)
  }
  def read_data(): (DenseMatrix[Double], DenseVector[Double], DenseMatrix[Double]) = {
    var source = scala.io.Source.fromFile("/home/mayer/LocalRepository/IdeaProjects/LinearRegression/src/main/scala/x.csv")
    var data = source.getLines.map(tmp => tmp.split(",")).toArray.map(_.map(_.toDouble))
    source.close
    val X = DenseMatrix(data:_*)

    source = scala.io.Source.fromFile("/home/mayer/LocalRepository/IdeaProjects/LinearRegression/src/main/scala/y.csv")
    val data_vector = source.getLines.toArray.map(_.toDouble)
    source.close
    val y = DenseVector(data_vector)

    source = scala.io.Source.fromFile("/home/mayer/LocalRepository/IdeaProjects/LinearRegression/src/main/scala/test.csv")
    data = source.getLines.map(tmp => tmp.split(",")).toArray.map(_.map(_.toDouble))
    source.close
    val test = DenseMatrix(data:_*)

    (X,y,test)
  }
}