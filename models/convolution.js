// 참고 자료
// 1. https://distill.pub/2016/deconv-checkerboard/
// 2. https://metamath1.github.io/cnn/index.html
// 3. https://ratsgo.github.io/deep%20learning/2017/04/05/CNNbackprop/
// 4. https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
class Convolution {
    constructor(filterSize, stride, numFilters, padding = 1, preTrainFilters = []) { // width, height, 
        // this.inputSize = inputSize;
        // this.width = width;
        // this.height = height;
        this.filterSize = filterSize;
        this.stride = stride;
        this.numFilters = numFilters;
        this.padding = padding;
        this.filters = [];
        if(preTrainFilters.length > 0){
            // this.filters = preTrainFilters;
            for (let i = 0; i < numFilters; i++) {
                // Initialize the filters randomly
                this.filters.push(new Array(filterSize).fill(0).map((v, j) => 
                    new Float32Array(filterSize).fill(0).map((v, k) => preTrainFilters[i][j][k])
                    // new Float32Array(filterSize).map((v, i) => (i % 3) - 1)
                ));
            }
        }else{
            for (let i = 0; i < numFilters; i++) {
                // Initialize the filters randomly
                this.filters.push(new Array(filterSize).fill(0).map(() => 
                    new Float32Array(filterSize).fill(0).map(() => Math.random() - 0.5)
                    // new Float32Array(filterSize).map((v, i) => (i % 3) - 1)
                ));
            }
        }
    }

    static load(info){
        console.log("convolution", info)
        return new Convolution(info.filterSize, info.stride, info.numFilters, info.padding, info.filters)
    }

    get_convolution_data(){
        return {
            filterSize:this.filterSize,
            stride:this.stride,
            numFilters:this.numFilters,
            padding:this.padding,
            filters:this.filters
        }
    }

    getImageData(image){
        const r = image.data.filter((v, i) => i % 4 == 0);
        const g = image.data.filter((v, i) => i % 4 == 1);
        const b = image.data.filter((v, i) => i % 4 == 2);

        return {r, g, b, width:image.width, height:image.height}
    }

    forward(image) {
        const {r, g, b, width, height} = this.getImageData(image)
        this.lastInput = image;
        const outputSize = ((width - this.filterSize + 1 + this.padding * 2) * (height - this.filterSize + 1 + this.padding * 2)) / this.stride;
        // console.log(outputSize)
        const output_map = new Array(this.numFilters).fill(0).map(() => {
            return {
                data:new Uint8ClampedArray(outputSize * 4 / this.stride).fill(0),
                colorSpace:"srgb",
                height:(height - this.filterSize + 1 + this.padding * 2) / this.stride,
                width:(width - this.filterSize + 1 + this.padding * 2) / this.stride
            }
        });
        // console.log(output_map)
        for (let f = 0; f < this.numFilters; f++) {
            this.conv(image.data, output_map[f], this.filters[f])
            this.maxpooling(output_map[f])
        }
        // console.log(output_map)
        return output_map;
    }

    conv(input, output, filters){
        for(let i = 0; i < output.height - this.padding; i++){
            for(let j = 0; j < output.width - this.padding; j++){
                const index = i * output.width + j;
                // console.log("image", i, j, index);
                for(let color = 0; color < 4; color++){ // rgba
                    const target_index = index * 4 + color;
                    let sum = 0;
                    // console.log("color", target_index)
                    for(let h = 0; h < filters.length; h++){ // f height
                        for(let w = 0; w < filters[h].length; w++){ // f width
                            const inner_index = (i + h) * this.stride * output.width * 4 * this.stride + ((j + w) * 4 * this.stride + color);
                            // console.log("inner_index", h, w, i + h, j + w, inner_index, input[inner_index], filters[h][w], sum);
                            // console.log(this.filters[l][h][w], input[i + w * h])
                            if(sum + filters[h][w] * input[inner_index] > 255){
                                sum = 255
                            }else{
                                // console.log(filters[h][w], input[i + w * h])
                                sum += filters[h][w] * input[inner_index];
                            }
                        }
                    }
                    if(color == 3){
                        // output.data[target_index] = output.data[target_index - 1] + output.data[target_index - 2] + output.data[target_index - 3] > 0?255:0;
                        output.data[target_index] = 255
                    }else{
                        output.data[target_index] = sum;
                    }
                }
            }
        }
    }

    maxpooling(output){
        if(output.height % 2 == 0 && output.width % 2 == 0){
            for(let i = 0; i < output.height; i += 2){
                for(let j = 0; j < output.width; j += 2){
                    for(let color = 0; color < 4; color++){ // rgba
                        const target_index = i * output.width + j * 2 + color;
                        // console.log("color", target_index)
                        let max = -1;
                        for(let h = 0; h < 2; h++){
                            for(let w = 0; w < 2; w++){
                                const inner_index = (i + h) * output.width * 4 + ((j + w) * 4 + color);
                                if(max < output.data[inner_index]){
                                    max = output.data[inner_index]
                                }
                                // console.log("inner_index", h, w, i + h, j + w, inner_index, output.data[inner_index], max);
                            }
                        }
                        output.data[target_index] = max;
                    }
                }
            }
            output.data = output.data.slice(0, output.data.length / 4)
            output.height /= 2;
            output.width /= 2;
        }
    }

    backward(d_L_d_out, learning_rate = 0.1) {
        let d_L_d_filters = [];
        for (let i = 0; i < this.numFilters; i++) {
            d_L_d_filters.push(new Array(this.filterSize).fill(0).map(() => new Float32Array(this.filterSize).fill(0)));
        }

        // de-conv
        let d_L_d_input = new Array(this.numFilters).fill(0).map(() => {
            return {
                data:new Uint8ClampedArray(this.lastInput.height * this.lastInput.width * 4).fill(0),
                colorSpace:"srgb",
                height:this.lastInput.height,
                width:this.lastInput.width
            }
        });

        for (let f = 0; f < this.numFilters; f++) {
            for(let i = 0; i < d_L_d_out[f].height; i++){
                for(let j = 0; j < d_L_d_out[f].width; j++){
                    for(let color = 0; color < 4; color++){ // rgba
                        const target_index = i * d_L_d_out[f].width * 4 + j * 4 + color;
                        // console.log("color", target_index)
                        for(let h = 0; h < 2; h++){ // f height
                            for(let w = 0; w < 2; w++){ // f width
                                const inner_index = (i * 2 + h) * this.lastInput.width * 4 + ((j * 2 + w) * 4 + color);
                                d_L_d_input[f].data[inner_index] = d_L_d_out[f].data[target_index];
                                // console.log("inner_index", h, w, i + h, j + w, inner_index, d_L_d_out[f].data[target_index], d_L_d_input[f][inner_index]);
                            }
                        }
                    }
                }
            }
            // console.log(d_L_d_input[f])
        }
        // 힘을 너무 많이 썻어ㅓ
        // filter value backward
        for (let f = 0; f < this.numFilters; f++) {
            for(let i = 0; i < d_L_d_input[f].height - this.filterSize + 1; i++){
                for(let j = 0; j < d_L_d_input[f].width - this.filterSize + 1; j++){
                    // console.log("image", i, j);
                    for(let color = 0; color < 4; color++){ // rgba
                        for(let h = 0; h < d_L_d_filters[f].length; h++){ // f height
                            for(let w = 0; w < d_L_d_filters[f][h].length; w++){ // f width
                                const inner_index = (i + h) * this.stride * d_L_d_input[f].width * 4 * this.stride + ((j + w) * 4 * this.stride + color);
                                if(inner_index >= d_L_d_input[f].data.length) break;
                                d_L_d_filters[f][h][w] += (d_L_d_input[f].data[inner_index] - this.lastInput.data[inner_index]) / (255 * (d_L_d_out[f].height - this.padding) * (d_L_d_out[f].width - this.padding) * 36);
                                // console.log("inner_index", h, w, i + h, j * 4 + w + color, inner_index, d_L_d_out[f].length, d_L_d_out[f].data[inner_index], this.lastInput[inner_index], d_L_d_filters[f][h][w], d_L_d_out[f].data[inner_index] * this.lastInput[inner_index]);
                            }
                        }
                    }
                }
            }
            // console.log(this.filters[f], d_L_d_filters[f])
        }

        // filter update
        for (let f = 0; f < this.numFilters; f++) {
            for (let i = 0; i < this.filterSize; i++) {
                for (let j = 0; j < this.filterSize; j++) {
                    if(Math.abs(d_L_d_filters[f][i][j]) > 10e-4){
                        this.filters[f][i][j] -= learning_rate * d_L_d_filters[f][i][j];
                    }
                }
            }
        }
        return d_L_d_input;
    }
}