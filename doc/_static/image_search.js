
function cosineSimilarity(vec1, vec2) {
    const dotProduct = vec1.map((val, i) => val * vec2[i]).reduce((accum, curr) => accum + curr, 0);
    const vec1Size = calcVectorSize(vec1);
    const vec2Size = calcVectorSize(vec2);
  
    return dotProduct / (vec1Size * vec2Size);
};

function calcVectorSize(vec) {
    return Math.sqrt(vec.reduce((accum, curr) => accum + Math.pow(curr, 2), 0));
};
  

fetch('/_static/data.json')
    .then( r => r.json() )
    .then( d => { 
        result = {}
        for (const [key, value] of Object.entries(d)) {
            console.log(`${key}: ${value}`);
            cos = cosineSimilarity(Object.values(d)[0], value)
            result[cos] = key
        }

        result = Object.keys(result).sort().reduce(
            (obj, key) => { 
              obj[key] = result[key]; 
              return obj;
            }, 
            {}
        );

        console.log(result)
    
        const container = document.getElementById('sphx-glr-imgsearchresult-container')
        container.innerHTML = ""
        console.log(container)
        Object.entries(result).map( ([key, value], index) => {
            if( index > 5 ) return
            console.log(value.split('/').at(-1), index)
            const id = `imgsearchref-(${value.split('/').at(-1)})`;
            const elem = document.getElementById( id );
            console.log(id)
            container.innerHTML += elem.innerHTML 
        } )

    } )

