class Neural{
    constructor(index){
        this.index = index;
        this.bias = Math.random() / 10; // 특성
        this.next_links = [];
        this.prev_links = [];
        this.sum = 0;
        this.output = 0; // 0 ~ 1
        this.error = 0;
        this.cost = 0;
    }

    static load(info){
        const node = new Neural(info.index)
        node.bias = info.bias; // 특성
        node.sum = info.sum;
        node.next_links_value = info.next_links_value;
        node.prev_links_value = info.prev_links_value;
        node.output = info.output; // 0 ~ 1
        node.error = info.error;
        node.cost = info.cost;
        return node;
    }

    activation(num){ // sum
        if(this.next_links.length == 0){
            // hyperbolic tangent
            let e2x = Math.exp(2 * num)
            if(num > 300){
                return 1
            }else if(num < -300){
                return -1;
            }else{
                if(isNaN(e2x)){
                    console.log("e2x", num, Math.exp(2 * num))
                    debugger;
                }
                return (e2x - 1) / (e2x + 1);
            }
        }
        // relu
        return num <= 0?0:num;
    }

    next_current(node, index = -1){
        if(index > -1){
            this.next_links.push(new Link(node, this.next_links_value[index]))
        }else{
            this.next_links.push(new Link(node))
        }
    }

    prev_current(node, index = -1){
        if(index > -1){
            this.prev_links.push(new Link(node, this.prev_links_value[index]))
        }else{
            this.prev_links.push(new Link(node))
        }
    }

    getLoss(target){
        // console.log("getLoss", 0.5 * Math.pow(this.output - target, 2))
        // 
        return 0.5 * Math.pow(this.output - target, 2)
    }

    difference(){
        // console.log("difference", this.sum)
        if(this.next_links.length == 0){
            // hyperbolic tangent gradiunt value
            return 1 - this.activation(this.output) ** 2
        }
        // relu
        return this.sum <= 0?0:1; // 0.001 * this.sum
    }

    backpropagateOutput(target, size) {
        // divide target size
        // this.cost = (this.output - target) / size;
        this.cost = (this.output - target) / size;
        // if(isNaN(this.cost)){
        //     console.log("cost", this.output, target, size)
        // }
        // console.log("backpropagateOutput", this.output, target, this.cost)
    }
    
    backpropagateHidden() {
        this.error = this.cost * this.difference();
        for (const link of this.prev_links) {
            if(!link.active) continue;
            link.error = this.error * link.node.output;
        }
    }
    
    updateWeights(learningRate) {
        for (const link of this.prev_links) {
            this.bias -= learningRate * this.error;
            if(!link.active) continue;
            link.weight -= learningRate * link.error;
        }
    }

    calculate(){
        this.sum = this.bias; // 고유 특성
        for(const link of this.prev_links){
            if(link.weight > -10e-6 && link.weight < 10e-6){
                link.active = false
                link.weight = 0;
                link.node.next_links[this.index].active = false
                link.node.next_links[this.index].weight = 0;
            }
            this.sum += link.weight * link.node.output;
        }
        // if(this.sum > 1e100 || isNaN(this.sum)){
        //     this.sum = 1e100
        // }else if(this.sum < -1e100){
        //     this.sum = -1e100
        // }
        this.output = this.activation(this.sum);
        if(isNaN(this.output)){
            console.log("sum", this.sum, this.index, this.activation(this.sum))
            debugger;
        }
        return this.output
    }
}