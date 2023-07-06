// accessible JavaScript tab switcher
// modified from https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Roles/Tab_Role

function getRandomInt(max) {
  return Math.floor(Math.random() * max);
}

var images_rotate = [
    {"image": "sphx_glr_plot_001_2_00x.png", "caption": "plot(x, y)", "link": "plot_types/basic/plot.html"},
    {"image": "sphx_glr_fill_between_001_2_00x.png", "caption": "fill_between(x, y1, y2)", "link": "plot_types/basic/fill_between.html"},
    {"image": "sphx_glr_scatter_plot_001_2_00x.png", "caption": "scatter(x, y)", "link": "plot_types/basic/scatter_plot.html"},
    {"image": "sphx_glr_pcolormesh_001_2_00x.png", "caption": "pcolormesh(X, Y, Z)", "link": "plot_types/arrays/pcolormesh.html"},
    {"image": "sphx_glr_contourf_001_2_00x.png", "caption": "contourf(X, Y, Z)", "link": "plot_types/arrays/contourf.html"},
    {"image": "sphx_glr_stairs_001_2_00x.png", "caption": "stairs(y)", "link": "plot_types/basic/stairs.html"},
    {"image": "sphx_glr_streamplot_001_2_00x.png", "caption": "streamplot(X, Y, U, V)", "link": "plot_types/arrays/streamplot.html"},
    {"image": "sphx_glr_bar_001_2_00x.png", "caption": "bar(x, height) / barh(y, width)", "link": "plot_types/basic/bar.html"},
    {"image": "sphx_glr_hist_plot_001_2_00x.png", "caption": "hist(x)", "link": "plot_types/stats/hist_plot.html"},
    {"image": "sphx_glr_imshow_001_2_00x.png", "caption": "imshow(Z)", "link": "plot_types/arrays/imshow.html"},
];

document.addEventListener("DOMContentLoaded", function(event) {
  ///////////////////////////////////////
  // rotate images in images-rotate directory:
  var ind = getRandomInt(images_rotate.length);
  var info = images_rotate[ind];
  var img_src = "../_images/" + info.image;
  var caption = info.caption;
  var link = "https://matplotlib.org/stable/" + info.link;
  var html = '<a href="' + link + '">' +
    '<img class="imrot-img" src="' + img_src + '" aria-labelledby="sample-plot-caption"/>' +
    '<div class="imrot-cap" id="sample-plot-caption">' + caption + '</div>' +
    '</a>';
document.getElementById('image_rotator').innerHTML = html;

  ind = getRandomInt(images_rotate.length);
  info = images_rotate[ind];
  img_src = "../_images/" + info.image;
  caption = info.caption;
  link = "https://matplotlib.org/stable/" + info.link;
  html = '<a href="' + link + '">' +
  '<img class="imrot-img" src="' + img_src + '" aria-labelledby="sample-plot-caption"/>' +
  '<div class="imrot-cap" id="sample-plot-caption">' + caption + '</div>' +
  '</a>';
document.getElementById('image_rotator2').innerHTML = html;

});
