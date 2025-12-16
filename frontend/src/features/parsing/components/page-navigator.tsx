import { LeftOutlined, RightOutlined } from "@ant-design/icons";
import "./page-navigator.css";

interface PageNavigatorProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
}

export function PageNavigator({
  currentPage,
  totalPages,
  onPageChange,
}: PageNavigatorProps) {
  const handlePrev = () => {
    if (currentPage > 1) {
      onPageChange(currentPage - 1);
    }
  };

  const handleNext = () => {
    if (currentPage < totalPages) {
      onPageChange(currentPage + 1);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value, 10);
    if (!isNaN(value) && value >= 1 && value <= totalPages) {
      onPageChange(value);
    }
  };

  return (
    <div className="page-navigator">
      <button
        className="page-navigator__btn"
        onClick={handlePrev}
        disabled={currentPage <= 1}
        aria-label="이전 페이지"
      >
        <LeftOutlined />
      </button>

      <div className="page-navigator__info">
        <input
          type="number"
          className="page-navigator__input"
          value={currentPage}
          onChange={handleInputChange}
          min={1}
          max={totalPages}
        />
        <span className="page-navigator__separator">/</span>
        <span className="page-navigator__total">{totalPages}</span>
      </div>

      <button
        className="page-navigator__btn"
        onClick={handleNext}
        disabled={currentPage >= totalPages}
        aria-label="다음 페이지"
      >
        <RightOutlined />
      </button>

      <div className="page-navigator__thumbnails">
        {Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => (
          <button
            key={page}
            className={`page-navigator__thumb ${
              page === currentPage ? "page-navigator__thumb--active" : ""
            }`}
            onClick={() => onPageChange(page)}
            aria-label={`페이지 ${page}`}
          />
        ))}
      </div>
    </div>
  );
}
