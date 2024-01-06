class Link {
    constructor(node, {weight =  0.5 - Math.random(), error = 0, active = true} = {}){
        this.node = node;
        this.weight = weight; // -0.5 ~ 0.5
        this.error = error;
        this.active = active;
    }
}