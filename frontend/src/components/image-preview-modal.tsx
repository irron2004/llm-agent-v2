import { useEffect, useCallback, useState } from "react";
import { LeftOutlined, RightOutlined, CloseOutlined, CheckOutlined } from "@ant-design/icons";
import { MarkdownContent } from "../features/chat/components/markdown-content";
import "./image-preview-modal.css";

export interface ImagePreviewItem {
  url?: string;           // 이미지 URL (없으면 텍스트 문서)
  content?: string;       // 텍스트 콘텐츠 (이미지가 없을 때 사용)
  title?: string;
  page?: number;
  docId?: string;
  rank?: number;
}

interface ImagePreviewModalProps {
  visible: boolean;
  images: ImagePreviewItem[];
  currentIndex: number;
  selectedRanks?: number[];           // 선택 모드용 (optional)
  onIndexChange: (index: number) => void;
  onClose: () => void;
  onToggleSelect?: (rank: number) => void;  // 선택 모드용 (optional)
}

export function ImagePreviewModal({
  visible,
  images,
  currentIndex,
  selectedRanks,
  onIndexChange,
  onClose,
  onToggleSelect,
}: ImagePreviewModalProps) {
  const currentImage = images[currentIndex];
  const isSelectionMode = Boolean(onToggleSelect && selectedRanks);
  const isCurrentSelected = currentImage?.rank !== undefined && selectedRanks?.includes(currentImage.rank);
  const hasImageUrl = Boolean(currentImage?.url);

  // 이미지 로드 실패 상태 추적
  const [imageLoadFailed, setImageLoadFailed] = useState(false);

  // 인덱스 변경 시 이미지 로드 실패 상태 초기화
  useEffect(() => {
    setImageLoadFailed(false);
  }, [currentIndex]);

  // 실제로 이미지를 표시할지 결정 (URL이 있고 로드 실패하지 않은 경우)
  const shouldShowImage = hasImageUrl && !imageLoadFailed;

  const handlePrev = useCallback(() => {
    if (currentIndex > 0) {
      onIndexChange(currentIndex - 1);
    }
  }, [currentIndex, onIndexChange]);

  const handleNext = useCallback(() => {
    if (currentIndex < images.length - 1) {
      onIndexChange(currentIndex + 1);
    }
  }, [currentIndex, images.length, onIndexChange]);

  const handleToggleSelect = useCallback(() => {
    if (onToggleSelect && currentImage?.rank !== undefined) {
      onToggleSelect(currentImage.rank);
    }
  }, [onToggleSelect, currentImage?.rank]);

  // 키보드 이벤트 처리
  useEffect(() => {
    if (!visible) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case "ArrowLeft":
          e.preventDefault();
          handlePrev();
          break;
        case "ArrowRight":
          e.preventDefault();
          handleNext();
          break;
        case "Escape":
          e.preventDefault();
          onClose();
          break;
        case " ":
          if (isSelectionMode) {
            e.preventDefault();
            handleToggleSelect();
          }
          break;
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [visible, handlePrev, handleNext, onClose, isSelectionMode, handleToggleSelect]);

  // 실제로 모달이 표시될 조건
  const shouldRender = visible && currentImage;

  // 모달 열릴 때 body 스크롤 막기
  useEffect(() => {
    if (shouldRender) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [shouldRender]);

  if (!shouldRender) return null;

  return (
    <div className="image-preview-modal-overlay" onClick={onClose}>
      <div className="image-preview-modal" onClick={(e) => e.stopPropagation()}>
        {/* 헤더 */}
        <div className="image-preview-modal-header">
          <button
            className="image-preview-modal-close"
            onClick={onClose}
            aria-label="닫기"
          >
            <CloseOutlined />
          </button>
          <span className="image-preview-modal-counter">
            {currentIndex + 1} / {images.length}
          </span>
        </div>

        {/* 메인 콘텐츠 영역 */}
        <div className="image-preview-modal-body">
          <button
            className="image-preview-modal-nav prev"
            onClick={handlePrev}
            disabled={currentIndex === 0}
            aria-label="이전"
          >
            <LeftOutlined />
          </button>

          <div className="image-preview-modal-image-container">
            {shouldShowImage ? (
              <img
                src={currentImage.url}
                alt={`${currentImage.title || "Document"} page ${currentImage.page || ""}`}
                className="image-preview-modal-image"
                onError={() => setImageLoadFailed(true)}
              />
            ) : (
              <div className="image-preview-modal-text-content">
                <MarkdownContent content={currentImage.content || ""} />
              </div>
            )}
          </div>

          <button
            className="image-preview-modal-nav next"
            onClick={handleNext}
            disabled={currentIndex === images.length - 1}
            aria-label="다음"
          >
            <RightOutlined />
          </button>
        </div>

        {/* 푸터 */}
        <div className="image-preview-modal-footer">
          <div className="image-preview-modal-info">
            <span className="image-preview-modal-title">
              {currentImage.title || "문서"}
            </span>
            {currentImage.page && (
              <span className="image-preview-modal-page">
                p.{currentImage.page}
              </span>
            )}
          </div>

          {isSelectionMode && (
            <button
              className={`image-preview-modal-select ${isCurrentSelected ? "selected" : ""}`}
              onClick={handleToggleSelect}
            >
              {isCurrentSelected ? (
                <>
                  <CheckOutlined /> 선택됨
                </>
              ) : (
                "이 문서 선택"
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
