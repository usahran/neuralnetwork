// https://towardsdatascience.com/loss-functions-and-their-use-in-neural-networks-a470e703f1e9
class Neural{
    constructor(index){
        this.index = index;
        this.bias = Math.random() / 10; // 특성
        this.next_links = []; // 다음 레이어 링크
        this.prev_links = []; // 이전 레이어 링크
        this.sum = 0; // 누적 합
        this.output = 0; // 0 ~ 1
        this.error = 0; // 활성 함수로 정규화된 오차 값
        this.cost = 0; // 일반적인 오차 값
    }

    static load(info){
        const node = new Neural(info.index)
        node.bias = info.bias; // 특성
        node.sum = info.sum; // 누적 합
        node.next_links_value = info.next_links_value; // 다음 레이어 링크
        node.prev_links_value = info.prev_links_value; // 이전 레이어 링크
        node.output = info.output; // 0 ~ 1 or -1 ~ 1
        node.error = info.error;
        node.cost = info.cost;
        return node;
    }

    activation(num){ // sum
        // 원래 히든 레이어도 모두 hyperbolic tangent였지만 infinite 값이 나오는 문제 있어 수정
        // 그리고 효율이 그닥 좋지 않음
        // 출력 부분 외 활성 함수는 relu로 수정
        // 출력값 정규화
        if(this.next_links.length == 0){
            // hyperbolic tangent
            let e2x = Math.exp(2 * num)
            if(num > 300){
                return 1
            }else if(num < -300){
                return -1;
            }else{
                if(isNaN(e2x)){ // 왜? 2
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
        // 왜 이런 식이 사용되는가 -> 출력이 1010이고 목표값이 1000이라면 1%내외의 편차를 가지며,
        // 이는 원하는 결과에 가깝다는 결론으로 볼 수 있습니다
        // 그런데 보다 정밀한 측정을 위해 오차를 제곱하여 음수 오차와 양수 오차의 상쇄를 막기 위함이며,
        // 제곱 미분 편의성을 위한 행동으로 해석됩니다 (n^2 미분 -> 2n이 되니 1/2과 상쇄)
        // https://builtin.com/machine-learning/loss-functions
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
        if(isNaN(this.output)){ // 왜? 1
            console.log("sum", this.sum, this.index, this.activation(this.sum))
            debugger;
        }
        return this.output
    }
}