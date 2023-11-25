class Neural{
    constructor(){
        this.bias = Math.random();
        this.next_links = [];
        this.prev_links = [];
        this.sum = 0;
        this.output = 0;
        this.error = 0;
        this.cost = 0;
    }
    // hyperbolic tangent
    activation(num){
        if(num > 1000000){
            return 1
        }else if(num < -1000000){
            return -1;
        }else{
            let e2x = Math.exp(2 * num)
            return (e2x - 1) / (e2x + 1);
        }
    }

    next_current(node){
        this.next_links.push(new Link(node))
    }

    prev_current(node){
        this.prev_links.push(new Link(node))
    }

    getLoss(target){
        return 0.5 * Math.pow(this.output - target, 2)
    }

    difference(){
        // hyperbolic tangent gradiunt value
        return 1 - this.activation(this.output) ** 2
    }

    backpropagateOutput(target, size) {
        // divide target size
        this.cost = (this.output - target) / size;
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
        this.sum = 0;
        for(const link of this.prev_links){
            if(link.weight > -0.00001 && link.weight < 0.00001)link.active = false;
            this.sum += link.weight * link.node.output + this.bias;
        }
        this.output = this.activation(this.sum);
        return this.output
    }
}