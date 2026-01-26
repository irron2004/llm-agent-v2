import { useState, useCallback } from "react";
import { StarOutlined, StarFilled } from "@ant-design/icons";

type RatingStarsProps = {
  value: number;
  onChange: (value: number) => void;
  label: string;
  disabled?: boolean;
  size?: "small" | "medium" | "large";
};

export function RatingStars({
  value,
  onChange,
  label,
  disabled = false,
  size = "medium",
}: RatingStarsProps) {
  const [hoverValue, setHoverValue] = useState<number | null>(null);

  const handleClick = useCallback(
    (star: number) => {
      if (disabled) return;
      onChange(star);
    },
    [disabled, onChange]
  );

  const handleMouseEnter = useCallback(
    (star: number) => {
      if (disabled) return;
      setHoverValue(star);
    },
    [disabled]
  );

  const handleMouseLeave = useCallback(() => {
    setHoverValue(null);
  }, []);

  const displayValue = hoverValue ?? value;

  const sizeStyles = {
    small: { fontSize: 14, gap: 2 },
    medium: { fontSize: 18, gap: 4 },
    large: { fontSize: 24, gap: 6 },
  };

  const { fontSize, gap } = sizeStyles[size];

  return (
    <div className="rating-stars-container">
      <label
        className="rating-stars-label"
        style={{
          display: "block",
          marginBottom: 4,
          fontSize: 12,
          color: "var(--color-text-secondary)",
          fontWeight: 500,
        }}
      >
        {label}
      </label>
      <div
        className="rating-stars"
        style={{
          display: "flex",
          alignItems: "center",
          gap: gap,
        }}
        onMouseLeave={handleMouseLeave}
      >
        {[1, 2, 3, 4, 5].map((star) => {
          const filled = star <= displayValue;
          return (
            <button
              key={star}
              type="button"
              onClick={() => handleClick(star)}
              onMouseEnter={() => handleMouseEnter(star)}
              disabled={disabled}
              style={{
                background: "none",
                border: "none",
                padding: 0,
                cursor: disabled ? "not-allowed" : "pointer",
                color: filled
                  ? "var(--color-accent-primary, #1890ff)"
                  : "var(--color-border, #d9d9d9)",
                fontSize: fontSize,
                lineHeight: 1,
                transition: "color 0.2s, transform 0.1s",
                transform: hoverValue === star ? "scale(1.1)" : "scale(1)",
                opacity: disabled ? 0.5 : 1,
              }}
              title={`${star}점`}
              aria-label={`${label} ${star}점`}
            >
              {filled ? <StarFilled /> : <StarOutlined />}
            </button>
          );
        })}
        <span
          style={{
            marginLeft: 8,
            fontSize: 12,
            color: "var(--color-text-secondary)",
          }}
        >
          {value > 0 ? `${value}점` : "선택 안함"}
        </span>
      </div>
    </div>
  );
}

export default RatingStars;
