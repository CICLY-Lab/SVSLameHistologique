window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


/* =============================================
   LIGHTBOX ZOOM
   ============================================= */

function initLightbox() {
  var overlay = null;
  var viewport = null;
  var img = null;
  var scale = 1;
  var translateX = 0;
  var translateY = 0;
  var panning = false;
  var panStart = { x: 0, y: 0 };
  var posStart = { x: 0, y: 0 };

  function open(src) {
    if (overlay) close();

    overlay = document.createElement('div');
    overlay.className = 'lb-overlay';

    viewport = document.createElement('div');
    viewport.className = 'lb-viewport';

    img = document.createElement('img');
    img.src = src;
    img.alt = '';

    var hint = document.createElement('div');
    hint.className = 'lb-hint';
    hint.textContent = 'Molette pour zoomer • Clic-glisser pour déplacer • Échap pour fermer';

    var controls = document.createElement('div');
    controls.className = 'lb-controls';

    var btnIn = document.createElement('button');
    btnIn.className = 'lb-btn';
    btnIn.textContent = '+';
    btnIn.title = 'Zoomer';

    var btnOut = document.createElement('button');
    btnOut.className = 'lb-btn';
    btnOut.textContent = '\u2212';
    btnOut.title = 'Dézoomer';

    var btnReset = document.createElement('button');
    btnReset.className = 'lb-btn';
    btnReset.textContent = '\u21BA';
    btnReset.title = 'Réinitialiser';

    var btnClose = document.createElement('button');
    btnClose.className = 'lb-btn';
    btnClose.textContent = '\u00D7';
    btnClose.title = 'Fermer';

    controls.appendChild(btnIn);
    controls.appendChild(btnOut);
    controls.appendChild(btnReset);
    controls.appendChild(btnClose);

    overlay.appendChild(viewport);
    overlay.appendChild(hint);
    overlay.appendChild(controls);
    viewport.appendChild(img);
    document.body.appendChild(overlay);
    document.body.style.overflow = 'hidden';

    resetTransform();

    btnIn.addEventListener('click', function(e) { e.stopPropagation(); zoom(1.3); });
    btnOut.addEventListener('click', function(e) { e.stopPropagation(); zoom(1 / 1.3); });
    btnReset.addEventListener('click', function(e) { e.stopPropagation(); resetTransform(); });
    btnClose.addEventListener('click', function(e) { e.stopPropagation(); close(); });

    overlay.addEventListener('click', function(e) {
      if (e.target === overlay) close();
    });

    viewport.addEventListener('wheel', function(e) {
      e.preventDefault();
      var dir = e.deltaY < 0 ? 1.3 : 1 / 1.3;
      zoom(dir, e.clientX, e.clientY);
    }, { passive: false });

    viewport.addEventListener('mousedown', function(e) {
      if (e.button !== 0) return;
      panning = true;
      viewport.classList.add('grabbing');
      panStart = { x: e.clientX, y: e.clientY };
      posStart = { x: translateX, y: translateY };
      e.preventDefault();
    });

    window.addEventListener('mousemove', function(e) {
      if (!panning) return;
      translateX = posStart.x + (e.clientX - panStart.x);
      translateY = posStart.y + (e.clientY - panStart.y);
      applyTransform();
    });

    window.addEventListener('mouseup', function() {
      if (panning) {
        panning = false;
        if (viewport) viewport.classList.remove('grabbing');
      }
    });

    window.addEventListener('keydown', onKey);
  }

  function onKey(e) {
    if (e.key === 'Escape') close();
  }

  function close() {
    if (overlay) {
      overlay.remove();
      overlay = null;
      viewport = null;
      img = null;
      scale = 1;
      translateX = 0;
      translateY = 0;
      document.body.style.overflow = '';
      window.removeEventListener('keydown', onKey);
    }
  }

  function resetTransform() {
    scale = 1;
    translateX = 0;
    translateY = 0;
    applyTransform();
  }

  function zoom(factor, cx, cy) {
    scale = Math.max(0.5, Math.min(20, scale * factor));

    if (cx !== undefined && cy !== undefined && viewport) {
      var rect = viewport.getBoundingClientRect();
      var px = cx - rect.left;
      var py = cy - rect.top;
      translateX = px - (factor) * (px - translateX);
      translateY = py - (factor) * (py - translateY);
    }

    applyTransform();
  }

  function applyTransform() {
    if (img) {
      img.style.transform = 'translate(' + translateX + 'px, ' + translateY + 'px) scale(' + scale + ')';
    }
  }

  document.querySelectorAll('.zoomable img, .comp-card img, .lb-enabled').forEach(function(el) {
    el.addEventListener('click', function(e) {
      var target = e.target.closest('.zoomable, .comp-card, .lb-enabled');
      if (!target) return;
      var imgs = target.querySelectorAll('img');
      if (imgs.length === 0) return;
      e.preventDefault();
      e.stopPropagation();
      open(imgs[0].src);
    });
  });
}


/* =============================================
   DOCUMENT READY
   ============================================= */

$(document).ready(function() {
  $(".navbar-burger").click(function() {
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });

  var options = {
    slidesToScroll: 1,
    slidesToShow: 3,
    loop: true,
    infinite: true,
    autoplay: false,
    autoplaySpeed: 3000,
  };

  var carousels = bulmaCarousel.attach('.carousel', options);

  for (var i = 0; i < carousels.length; i++) {
    carousels[i].on('before:show', function(state) {
      console.log(state);
    });
  }

  var element = document.querySelector('#my-element');
  if (element && element.bulmaCarousel) {
    element.bulmaCarousel.on('before-show', function(state) {
      console.log(state);
    });
  }

  preloadInterpolationImages();

  $('#interpolation-slider').on('input', function(event) {
    setInterpolationImage(this.value);
  });
  setInterpolationImage(0);
  $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

  bulmaSlider.attach();

  initLightbox();
});
