class Neuralnetwork {
    constructor(size = [], learningRate = 0.1){
        this.size = size;
        this.input = [];
        this.learningRate = learningRate;
        this.init()
    }

    init(){
        this.network = []
        this.create_node()
        this.current_node()
    }

    create_node(){
        for(let i = 0; i < this.size.length; i++){
            this.network.push([])
            for(let j = 0; j < this.size[i]; j++){
                this.network[this.network.length - 1].push(new Neural())
            }
        }
    }

    current_node(){
        for(let i = 0; i < this.network.length - 1; i++){
            for(let j = 0; j < this.network[i].length; j++){
                for(let k = 0; k < this.network[i + 1].length; k++){
                    this.network[i + 1][k].prev_current(this.network[i][j])
                    this.network[i][j].next_current(this.network[i + 1][k])
                }
            }
        }
    }

    forward(input = []){
        if(input.length > 0){
            this.input = input
        }
        for(let i = 0; i < this.network.length; i++){
            for(let j = 0; j < this.network[i].length; j++){
                if(i == 0){
                    this.network[0][j].output = this.input[j];
                }else{
                    this.network[i][j].calculate()
                }
            }
        }
        return this.network[this.network.length - 1];
    }

    backward(targets) {
        for (let i = 0; i < this.network[this.network.length - 1].length; i++) {
            this.network[this.network.length - 1][i].backpropagateOutput(targets[i], targets.length);
        }
    
        for (let i = this.network.length - 1; i > 0; i--) {
            for (const node of this.network[i]) {
                node.backpropagateHidden();
            }
            if(i > 0){
                let prevLayer = this.network[i - 1];
                for (let j = 0; j < prevLayer.length; j++) {
                    let node = prevLayer[j];
                    node.cost = 0;
                    for (const link of node.next_links) {
                        node.cost += link.weight * link.node.error;
                    }
                }
            }
        }
        this.updateWeights();
    }

    updateWeights() {
        for (let layer of this.network) {
            for (const node of layer) {
                node.updateWeights(this.learningRate);
            }
        }
    }

    getLoss(targets){
        let loss = 0;
        for (let i = 0; i < this.network[this.network.length - 1].length; i++) {
            // console.log("output", targets[i], this.network[this.network.length - 1][i].output, this.network[this.network.length - 1][i].getLoss(targets[i]))
            loss += this.network[this.network.length - 1][i].getLoss(targets[i]);
        }
        return loss;
    }

    train(inputs, targets) {
        const loop = 10;
        for(let i = 0; i < loop; i++){
            let loss = 0;
            for(const j in inputs){
                this.input = inputs[j];
                this.forward();
                this.backward(targets[j]);
                this.updateWeights();
                loss += this.getLoss(targets[j])
            }
            if(loss < 0.001){
                return loss
            }
        }
    }

    network_output(){
        return this.network[this.network.length - 1].map(v => v.output)
    }

    draw(ctx, width, height, size = 18){
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        var layerSpacing = width / (this.network.length + 1);
        var neuronSpacing = height / (Math.max(...this.network.map(v => v.length)) + 1);
        const rad = size / 2;
        ctx.lineWidth = 1;
        for(let i = 0; i < this.network.length; i++){
            for(let j = 0; j < this.network[i].length; j++){
                // console.log(this.network[i][j].output)
                const color = `rgba(${(this.network[i][j].output + (i == 0?0:1)) / 2 * 255}, ${((1 + (i == 0?1:0) - this.network[i][j].output) / 2) * 255}, 255, 1.0)`
                ctx.beginPath();
                var x = (i + 1) * layerSpacing;
                var y = (j + 1) * neuronSpacing;
                ctx.fillStyle = color;
                ctx.arc(x, y, rad, 0, Math.PI * 2);
                ctx.stroke();
                ctx.fill()
                for (let k = 0; k < this.network[i][j].prev_links.length; k++) {
                    const color = `rgba(${(this.network[i][j].prev_links[k].weight + 1) / 2 * 255}, ${((1 - this.network[i][j].prev_links[k].weight) / 2) * 255}, 255, 1.0)`
                    ctx.lineWidth = Math.round(this.network[i][j].prev_links[k].weight + 1) ;
                    var prevX = i * layerSpacing;
                    var prevY = (k + 1) * neuronSpacing;
                    // console.log(x, y, prevX, prevY, color)
                    ctx.fillStyle = this.network[i][j].prev_links[k].active?color:"#000";
                    ctx.strokeStyle = this.network[i][j].prev_links[k].active?color:"#000";
                    ctx.moveTo(prevX, prevY);
                    ctx.lineTo(x, y);
                    ctx.stroke();
                }
            }
        }
    }
}