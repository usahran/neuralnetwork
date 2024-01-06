const fs = require("fs")
let dataFileBuffer  = fs.readFileSync(__dirname + '/train-images.idx3-ubyte');
let labelFileBuffer = fs.readFileSync(__dirname + '/train-labels.idx1-ubyte');
let pixelValues     = [];
for (let image = 0; image < 40000; image++) {
    // let rand_index = (Math.round(Math.random() * 25000));
    let pixels = [];

    for (let x = 0; x < 28; x++) {
        for (let y = 0; y < 28; y++) {
            let pixel = dataFileBuffer[(image * 28 * 28) + (y + (x * 28)) + 15]
            pixels.push([pixel, pixel, pixel, 255]);
        }
    }

    let imageData = {};
    imageData[JSON.stringify(labelFileBuffer[image + 8])] = pixels.flat();

    pixelValues.push(imageData);
}
console.log(pixelValues)
try{
    fs.writeFileSync("data.js", "const images = " + JSON.stringify(pixelValues) + "\n", "utf-8");
}catch(e){
    console.log(e)
    fs.writeFileSync("data.js", pixelValues, "utf-8");
}
