function cosineSimilarity(vec1, vec2) {
    const dotProduct = vec1.map((val, i) => val * vec2[i]).reduce((accum, curr) => accum + curr, 0);
    const vec1Size = calcVectorSize(vec1);
    const vec2Size = calcVectorSize(vec2);
  
    return dotProduct / (vec1Size * vec2Size);
};

function calcVectorSize(vec) {
    return Math.sqrt(vec.reduce((accum, curr) => accum + Math.pow(curr, 2), 0));
};


data = []
fetch('/_static/data.json')
    .then( r => r.json() )
    .then( d => { data = d } )   


function handle_search() {
    if( data.length == 0 ){
        return;
    }

    const container = document.getElementById('sphx-glr-imgsearchresult-container')
    container.innerHTML = ""

    result = {}
    for (const [key, value] of data ) {
        // just find the similar images to the image at the beginning of data
        cos = cosineSimilarity( data[0][1], value)
        result[cos] = key
    }

    result = Object.keys(result).sort().reduce(
        (obj, key) => { 
            obj[key] = result[key]; 
            return obj;
        }, 
        {}
    );

    
    Object.entries(result).map( ([key, value], index) => {
        if( index > 5 ) return
        const id = value;
        const elem = document.getElementById( id );
        container.innerHTML += elem.innerHTML 
    } )

}

window.addEventListener( 'load', () => {
    document.getElementById('sphx-glr-imgsearchbutton').addEventListener( 'click', handle_search )
} )