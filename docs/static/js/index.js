// Mobile burger menu toggle (Bulma)
document.addEventListener('DOMContentLoaded', function () {
  const burgers = Array.prototype.slice.call(document.querySelectorAll('.navbar-burger'), 0);
  burgers.forEach(function (el) {
    el.addEventListener('click', function () {
      const target = el.dataset.target;
      const menu = document.getElementById(target);
      el.classList.toggle('is-active');
      if (menu) menu.classList.toggle('is-active');
    });
  });
});
