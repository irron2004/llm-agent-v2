import { Link } from "react-router-dom";
import { Result, Button } from "antd";

export default function NotFoundPage() {
  return (
    <Result
      status="404"
      title="Page not found"
      subTitle="요청하신 페이지를 찾을 수 없습니다."
      extra={
        <Link to="/">
          <Button type="primary">Go home</Button>
        </Link>
      }
    />
  );
}
