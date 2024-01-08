class Network {
    // [3, 2, 3]
    constructor(size = [], learningRate = 0.1){
        this.size = size;
        this.input = [];
        this.network = []
        this.learningRate = learningRate;
    }

    // 저장된 네트워크 적용
    static load(info){
        const network = new Network(info.size, info.learningRate)
        network.init(info.info)
        return network;
    }

    init(neurals = []){
        this.create_node(neurals)
        this.current_node(neurals)
    }

    create_node(neurals = []){
        for(let i = 0; i < this.size.length; i++){
            this.network.push([])
            for(let j = 0; j < this.size[i]; j++){
                if(neurals.length > 0){
                    const info = neurals[i][j];
                    this.network[this.network.length - 1].push(Neural.load(info))
                }else{
                    this.network[this.network.length - 1].push(new Neural(j))
                }
            }
        }
    }

    current_node(neurals = []){
        for(let i = 0; i < this.network.length - 1; i++){
            for(let j = 0; j < this.network[i].length; j++){
                for(let k = 0; k < this.network[i + 1].length; k++){
                    if(neurals.length > 0){
                        this.network[i + 1][k].prev_current(this.network[i][j], j)
                        this.network[i][j].next_current(this.network[i + 1][k], k)
                    }else{
                        this.network[i + 1][k].prev_current(this.network[i][j])
                        this.network[i][j].next_current(this.network[i + 1][k])
                    }
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
        // 출력 값 cost 산정
        for (let i = 0; i < this.network[this.network.length - 1].length; i++) {
            this.network[this.network.length - 1][i].backpropagateOutput(targets[i], targets.length);
        }
    
        // 대체 언제 완성되는데?
        // 다음 레이어의 cost로 활성함수로 정규화된 값 error 변수 저장
        for (let i = this.network.length - 1; i > 0; i--) {
            for (const node of this.network[i]) {
                node.backpropagateHidden();
            }
            // 이전 레이어에 연결된 각
            let prevLayer = this.network[i - 1];
            for (let j = 0; j < prevLayer.length; j++) {
                let node = prevLayer[j];
                node.cost = 0;
                for (const link of node.next_links) {
                    node.cost += link.weight * link.node.error;
                }
            }
        }
        // this.updateWeights();
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
        const loop = 1;
        let loss = 0;
        for(let i = 0; i < loop; i++){
            for(const j in inputs){
                this.input = inputs[j];
                this.forward();
                this.backward(targets[j]);
                this.updateWeights();
                loss += this.getLoss(targets[j])
            }
            if((loss / inputs.length) < 0.001){
                return loss / inputs.length / (i + 1)
            }
        }
        return loss / inputs.length / loop;
    }

    network_output(){
        return this.network[this.network.length - 1].map(v => v.output)
    }

    get_network_data(){
        const network = []
        for(let i = 0; i < this.network.length; i++){
            network.push([])
            for(let j = 0; j < this.network[i].length; j++){
                const index = this.network[i][j].index
                const bias = this.network[i][j].bias
                const next_links = this.network[i][j].next_links;
                const prev_links = this.network[i][j].prev_links;
                const next_links_value = []
                const prev_links_value = []
                for(const link of next_links){
                    next_links_value.push({
                        weight:link.weight,
                        error:link.error,
                        active:link.active
                    })
                }
                for(const link of prev_links){
                    prev_links_value.push({
                        weight:link.weight,
                        error:link.error,
                        active:link.active
                    })
                }
                const sum = this.network[i][j].sum
                const output = this.network[i][j].output
                const error = this.network[i][j].error
                const cost = this.network[i][j].cost
                network[i].push({
                    index,
                    bias,
                    next_links_value,
                    prev_links_value,
                    sum,
                    output,
                    error,
                    cost
                })
            }
        }
        return network;
    }

    draw(ctx, width, height, size = 18){
        ctx.clearRect(0, 0, width, height);
        var layerSpacing = width / (this.network.length + 1);
        const rad = size / 2;
        ctx.lineWidth = 1;
        for(let i = 0; i < this.network.length; i++){
            var neuronSpacing = height / (this.network[i].length + 1);
            for(let j = 0; j < this.network[i].length; j++){
                // if(i == 0){
                //     console.log(this.network[i][j].output)
                // }
                // const color = `rgba(${(this.network[i][j].output + (i == 0?0:1)) / 2 * 255}, ${((1 + (i == 0?1:0) - this.network[i][j].output) / 2) * 255}, 255, 1.0)`
                const color = `rgba(${this.network[i][j].output * 255}, ${1 - this.network[i][j].output * 255}, 255, 1.0)`
                ctx.beginPath();
                var x = (i + 1) * layerSpacing;
                var y = j * neuronSpacing;
                ctx.fillStyle = color;
                ctx.strokeStyle = color;
                ctx.arc(x, y, rad, 0, Math.PI * 2);
                // ctx.stroke();
                ctx.fill()
                for (let k = 0; k < this.network[i][j].next_links.length; k++) {
                    const line_color = `rgba(${(this.network[i][j].next_links[k].weight + 1) / 2 * 255}, ${((1 - this.network[i][j].next_links[k].weight) / 2) * 255}, 255, 1.0)`
                    ctx.lineWidth = Math.round(this.network[i][j].next_links[k].weight + 1);
                    var inner_neuronSpacing = height / (this.network[i + 1].length + 1);
                    var prevX = (i + 2) * layerSpacing;
                    var prevY = k * inner_neuronSpacing;
                    // console.log(x, y, prevX, prevY, line_color, this.network[i][j].next_links[k])
                    ctx.fillStyle = this.network[i][j].next_links[k].active?line_color:"#000";
                    ctx.strokeStyle = this.network[i][j].next_links[k].active?line_color:"#000";
                    ctx.moveTo(x, y);
                    ctx.lineTo(prevX, prevY);
                    ctx.stroke();
                }
            }
        }
    }
}