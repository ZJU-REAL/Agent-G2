document.querySelectorAll("[data-carousel]").forEach((carousel) => {
  const carouselSection = carousel.parentElement;
  const track = carousel.querySelector(".carousel-track");
  const slides = Array.from(carousel.querySelectorAll("[data-carousel-slide]"));
  const dots = Array.from(carouselSection.querySelectorAll("[data-carousel-dot]"));
  const prevButton = carousel.querySelector("[data-carousel-prev]");
  const nextButton = carousel.querySelector("[data-carousel-next]");
  let activeIndex = slides.findIndex((slide) => slide.classList.contains("is-active"));

  if (!track || !slides.length || !prevButton || !nextButton) {
    return;
  }

  if (activeIndex < 0) {
    activeIndex = 0;
  }

  const showSlide = (nextIndex) => {
    activeIndex = (nextIndex + slides.length) % slides.length;
    track.style.transform = `translateX(-${activeIndex * 100}%)`;

    slides.forEach((slide, index) => {
      const isActive = index === activeIndex;
      slide.classList.toggle("is-active", isActive);
      slide.setAttribute("aria-hidden", String(!isActive));
    });

    dots.forEach((dot, index) => {
      const isActive = index === activeIndex;
      dot.classList.toggle("is-active", isActive);
      dot.setAttribute("aria-current", String(isActive));
    });
  };

  prevButton.addEventListener("click", () => showSlide(activeIndex - 1));
  nextButton.addEventListener("click", () => showSlide(activeIndex + 1));

  dots.forEach((dot, index) => {
    dot.addEventListener("click", () => showSlide(index));
  });

  carousel.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") {
      showSlide(activeIndex - 1);
    }

    if (event.key === "ArrowRight") {
      showSlide(activeIndex + 1);
    }
  });

  showSlide(activeIndex);
});

const revealItems = Array.from(document.querySelectorAll(".section > .container, .footer .container"));

if (revealItems.length && "IntersectionObserver" in window) {
  document.body.classList.add("reveal-ready");

  const revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          revealObserver.unobserve(entry.target);
        }
      });
    },
    {
      rootMargin: "0px 0px -12% 0px",
      threshold: 0.12,
    },
  );

  revealItems.forEach((item) => {
    item.classList.add("reveal-item");
    revealObserver.observe(item);
  });
}
