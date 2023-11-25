class Link {
    constructor(node){
        this.node = node;
        this.weight = 0.5 - Math.random();
        this.error = 0;
        this.active = true;
    }
}